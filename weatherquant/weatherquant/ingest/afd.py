"""AFD forecaster-disagreement flag (ING-07, D-13) — pre-filter + OpenAI forced tool-use.

The AFD is untrusted human free-text; a flagged disagreement is a soft Phase-4 sizing
modifier, never a hard gate. Two fixes over v3: a keyword pre-filter gates before any paid
call to stay in budget, and a forced ``tool_choice`` on a strict function ``parameters`` schema
returns a guaranteed-shape dict (bounding prompt-injection, T-02-08/ASVS V5) instead of decoding
a JSON text body. The result stores as an ``observations`` row with ``source='afd'`` via the
single audited writer; an unset ``OPENAI_API_KEY`` degrades to no-signal (D-06/D-10/D-11; see
docs/DECISIONS.md).
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, date, datetime
from typing import Any

import httpx

from weatherquant.db.engine import get_settings
from weatherquant.db.types import Bind
from weatherquant.ingest.errors import AvailabilityError
from weatherquant.ingest.sources._client import managed_client
from weatherquant.ingest.writer import insert_observation
from weatherquant.registry import CITIES

logger = logging.getLogger(__name__)

SOURCE = "afd"

# The exact model id: OpenAI gpt-5.4-nano, the cheapest current-gen tier matching the
# $3-8/day budget (~$0.18/mo at this codebase's tiny pre-filtered AFD load). Supports forced
# function calling for the guaranteed-shape tool result.
AFD_MODEL = "gpt-5.4-nano"

# Fixed external endpoint (SSRF guard T-02-11: WFO codes come from the static map below,
# never from untrusted input; the host is a constant).
_NWS_API_BASE = "https://api.weather.gov"
# A descriptive User-Agent is REQUIRED by api.weather.gov (v3 NWS_UA shape, L77).
_USER_AGENT = "weatherquant/0.1 (kalshi daily-high paper-trading)"


# --- v3 pre-filter, ported VERBATIM (kalshi_weather_bot_v3.py L277-309) -------------
# These two keyword lists and afd_should_classify gate BEFORE any paid OpenAI call so
# routine discussions never cost a token (D-13, budget). Copied byte-for-byte from v3.
AFD_ROUTINE_PHRASES = [
    "no significant", "no significant changes",
    "near normal", "close to normal", "around normal",
    "seasonal temperatures", "typical for this time",
    "quiet pattern", "tranquil", "uneventful",
    "high pressure", "dry and sunny", "dry weather",
    "dominated by high pressure",
]
AFD_SIGNAL_KEYWORDS = [
    "uncertainty", "uncertain",
    "warmer than", "cooler than",
    "above normal", "below normal",
    "well above", "well below",
    "warmer than expected", "cooler than expected",
    "model disagreement", "model spread", "model differences",
    "pattern change", "significant change",
    "temperature forecast", "high temperature concern",
    "confidence", "low confidence", "high confidence",
    "degrees warmer", "degrees cooler",
    "surge", "anomaly", "anomalous",
    "record", "exceptional",
]


def afd_should_classify(text: str) -> tuple[bool, str]:
    """Keyword pre-filter gating before any paid call — return ``(should_classify, reason)`` (D-13).

    A routine opening phrase in the first 300 chars short-circuits to ``False`` (no API call);
    else a signal keyword anywhere yields ``True``; absent both, ``False``.
    """
    opening = text[:300].lower()
    for phrase in AFD_ROUTINE_PHRASES:
        if phrase in opening:
            return False, f"routine opening: '{phrase}'"
    body = text.lower()
    for kw in AFD_SIGNAL_KEYWORDS:
        if kw in body:
            return True, f"signal keyword: '{kw}'"
    return False, "no signal keywords found"


# --- WFO map for the 7 registry cities ---------------------------------------------
# AFD is issued per NWS Forecast Office (WFO), NOT per city — ported from v3
# CITY_COORDS[...][4] and re-verified against weatherquant.registry.CITIES (the 7
# in-scope Kalshi cities). DAL/DC/SEA/etc. from v3 are out of scope for Phase 2.
CITY_WFO: dict[str, str] = {
    "NYC": "OKX",  # New York, NY
    "CHI": "LOT",  # Chicago, IL
    "AUS": "EWX",  # Austin/San Antonio, TX
    "MIA": "MFL",  # Miami, FL
    "LAX": "LOX",  # Los Angeles/Oxnard, CA
    "DEN": "BOU",  # Denver/Boulder, CO
    "PHI": "PHI",  # Philadelphia/Mount Holly
}

# Re-verify the WFO map covers exactly the registry cities (fail loud at import on drift).
assert set(CITY_WFO) == set(CITIES), (
    f"CITY_WFO {sorted(CITY_WFO)} must match registry cities {sorted(CITIES)}"
)


# --- OpenAI forced tool-use (Pattern 7) --------------------------------------------
# A single function tool whose `parameters` IS the JSON shape; forcing tool_choice returns a
# guaranteed-shape dict and bounds prompt-injection to those keys (D-13, T-02-08, ASVS V5).
_AFD_TOOL = {
    "type": "function",
    "function": {
        "name": "record_afd_signal",
        "description": "Record the forecaster-disagreement signal from an NWS AFD excerpt.",
        "parameters": {
            "type": "object",
            "properties": {
                "disagreement": {
                    "type": "boolean",
                    "description": "True if the forecaster flags model disagreement / low "
                    "confidence in tomorrow's high temperature.",
                },
                "direction": {
                    "type": "string",
                    "enum": ["warmer", "cooler", "uncertain", ""],
                    "description": "Lean of the disagreement vs guidance, or '' if none.",
                },
                "summary": {
                    "type": "string",
                    "description": "One-sentence summary of the temperature signal, or ''.",
                },
            },
            "required": ["disagreement", "direction", "summary"],
        },
    },
}

# The no-signal result returned without any API call (pre-filter skip / unset key / degrade).
_NO_SIGNAL = {"disagreement": False, "direction": "", "summary": ""}


def classify_afd(
    text: str,
    wfo: str,
    client: object | None = None,
) -> dict[str, Any]:
    """Classify one AFD excerpt into a structured disagreement signal (ING-07, D-13).

    Pre-filter first (no-signal text → no API call); unset key → no-signal dict (D-11); else
    force the ``record_afd_signal`` tool and return its ``input`` dict, never decoding the text
    body as JSON.

    Args:
        client: optional pre-built OpenAI client; the unit test injects a mock so no real
            paid call is made. ``None`` builds a real ``openai.OpenAI``.

    Returns:
        ``{"disagreement": bool, "direction": str, "summary": str}`` — always this shape.
    """
    if not text:
        return dict(_NO_SIGNAL)

    should, reason = afd_should_classify(text)
    if not should:
        # Pre-filter skip — NO API call (D-13, budget). Carry the reason for observability.
        return {**_NO_SIGNAL, "reason": reason}

    if client is None:
        api_key = get_settings().openai_api_key
        if not api_key:
            # Graceful degrade (D-11): the key is unset — log a structured skip, no call.
            logger.warning(
                "AFD classify skipped for WFO=%s: OPENAI_API_KEY unset (degrade, D-11)",
                wfo,
            )
            return {**_NO_SIGNAL, "reason": "openai_api_key unset"}
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

    try:
        completion = client.chat.completions.create(  # type: ignore[attr-defined]
            model=AFD_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": f"WFO: {wfo}\n\nAFD excerpt:\n{text[:1500]}",
                }
            ],
            tools=[_AFD_TOOL],
            tool_choice={"type": "function", "function": {"name": "record_afd_signal"}},
            max_completion_tokens=256,
        )
        # Forced tool_choice → the response's first choice carries exactly one tool call whose
        # `function.arguments` is a JSON STRING of the schema-shaped dict. We json.loads ONLY those
        # arguments — the model text body is never decoded as JSON (v3 fragility, D-13, T-02-08).
        # Parsing stays INSIDE the try: a truncated tool call (e.g. max_completion_tokens hit)
        # yields malformed JSON, and JSONDecodeError must degrade like any SDK error (D-11).
        tool_calls = completion.choices[0].message.tool_calls
        if not tool_calls:
            logger.warning("AFD classify for WFO=%s returned no tool call; degrading", wfo)
            return dict(_NO_SIGNAL)
        result = json.loads(tool_calls[0].function.arguments)
    except Exception as exc:  # noqa: BLE001 - any OpenAI/SDK/parse error degrades to no-signal (D-11).
        logger.warning(
            "AFD classify failed for WFO=%s (%s); degrading to no-signal (D-11)", wfo, exc
        )
        return {**_NO_SIGNAL, "reason": "openai_error"}

    # Defensive: guarantee a dict with the three required keys even if the model returns a
    # non-object (null/array) or the SDK changes.
    if not isinstance(result, dict):
        return dict(_NO_SIGNAL)
    return {
        "disagreement": bool(result.get("disagreement", False)),
        "direction": result.get("direction", ""),
        "summary": result.get("summary", ""),
    }


async def fetch_afd_text(
    wfo: str,
    client: httpx.AsyncClient | None = None,
) -> str | None:
    """Fetch the latest AFD product text for ``wfo`` from api.weather.gov (Pattern 7).

    Two-step: ``/products/types/AFD/locations/{wfo}`` → latest id → ``/products/{id}`` →
    ``productText``. Fixed host + static ``wfo`` map (no SSRF, T-02-11). ``None`` on error (D-11).
    """
    async with managed_client(client) as client:
        try:
            list_resp = await client.get(
                f"{_NWS_API_BASE}/products/types/AFD/locations/{wfo}",
                headers={"User-Agent": _USER_AGENT, "Accept": "application/geo+json"},
            )
            list_resp.raise_for_status()
            products = list_resp.json().get("@graph", [])
            if not products:
                return None
            latest_id = products[0].get("id")
            if not latest_id:
                return None
            prod_resp = await client.get(
                f"{_NWS_API_BASE}/products/{latest_id}",
                headers={"User-Agent": _USER_AGENT},
            )
            prod_resp.raise_for_status()
            text = prod_resp.json().get("productText", "")
            return text if isinstance(text, str) else ""
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("AFD fetch failed for WFO=%s (%s)", wfo, exc)
            return None


def store_afd_signal(
    bind: Bind,
    city: str,
    target_date: date,
    result: dict[str, Any],
    available_at: datetime | None = None,
    mode: str = "live",
) -> int:
    """Persist an AFD signal as an ``observations`` row via the audited writer (D-06/D-10/D-11).

    ``available_at`` MUST be the product issuance time. ``now(UTC)`` is permitted only in live;
    backfill REQUIRES an explicit ``available_at`` and otherwise raises, since ``now()`` on a
    historical row leaks (CR-01/D-09; see docs/DECISIONS.md).

    Returns:
        ``1`` if a row was inserted, ``0`` if an identical row already existed (skip).
    """
    detail = {
        "disagreement": bool(result.get("disagreement", False)),
        "direction": result.get("direction", ""),
        "summary": result.get("summary", ""),
    }
    if available_at is None:
        if mode == "backfill":
            # CR-01: never stamp now() on a backfilled row (the look-ahead leak, D-09). An
            # AvailabilityError (a CorrectnessError) so it fails loud through the orchestrator,
            # never a silent skip (WR-05). Still a ValueError.
            raise AvailabilityError(
                f"store_afd_signal in backfill requires an explicit available_at "
                f"(AFD issuance time) for city={city} target_date={target_date} — refusing to "
                f"stamp now() on a historical row (CR-01/D-09)"
            )
        # Live: now(UTC) is the instant the running system actually held the product (D-09).
        available_at = datetime.now(UTC)
    return insert_observation(
        bind,
        city=city,
        target_date=target_date,
        source=SOURCE,
        detail=detail,
        available_at=available_at,
    )


__all__ = [
    "AFD_MODEL",
    "AFD_ROUTINE_PHRASES",
    "AFD_SIGNAL_KEYWORDS",
    "CITY_WFO",
    "SOURCE",
    "afd_should_classify",
    "classify_afd",
    "fetch_afd_text",
    "store_afd_signal",
]
