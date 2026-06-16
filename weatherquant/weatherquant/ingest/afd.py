"""AFD forecaster-disagreement flag (ING-07, D-13) — pre-filter + Anthropic forced tool-use.

The NWS Area Forecast Discussion (AFD) is free-text written by a human forecaster. When the
forecaster flags model disagreement / low confidence in tomorrow's high, that is a soft
sizing modifier for Phase 4 (PRC-05) — NEVER a hard trade gate. Because the blast radius is
low and the input is untrusted free-text (a prompt-injection vector, T-02-08), the Anthropic
output is constrained to a fixed tool ``input_schema`` so injected instructions cannot escape
the structured shape (ASVS V5).

Two cost/robustness fixes over v3 (kalshi_weather_bot_v3.py):

1. **Pre-filter before any paid call (D-13, budget).** :func:`afd_should_classify` is ported
   VERBATIM from v3 (L277-309): a routine opening ("near normal", "high pressure", …) returns
   ``(False, reason)`` and NO Anthropic call is made; a signal keyword returns ``(True, reason)``
   and classification proceeds. This keeps per-city-per-cycle cost inside the $3-8/day budget.

2. **Forced ``tool_choice`` instead of parsing a JSON text body (D-13).** v3 decoded the
   message text as JSON (L921) — fragile, breaks on any prose wrapper. Here a single
   ``record_afd_signal`` tool with a strict ``input_schema`` is forced via
   ``tool_choice={"type":"tool","name":...}`` so the model returns a guaranteed-shape dict.

The result is stored as an ``observations`` row with ``source='afd'`` and ``detail`` jsonb =
``{disagreement, direction, summary}`` (D-06) — never a new table — through 02-02's single
audited :func:`weatherquant.ingest.writer.insert_observation` path (D-10/D-11). The
``ANTHROPIC_API_KEY`` is read from ``Settings`` (never the raw process environment inline;
redacted repr, T-02-09); when unset the classifier logs a structured skip and returns no-signal
(D-11) — the pre-filter still runs first either way.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import httpx

from weatherquant.db.engine import get_settings
from weatherquant.ingest.errors import AvailabilityError
from weatherquant.ingest.writer import insert_observation
from weatherquant.registry import CITIES

logger = logging.getLogger(__name__)

SOURCE = "afd"

# The exact model id, pinned to the claude-haiku-4-5 family (cheap tier matching the
# $3-8/day budget; v3 used claude-haiku-4-5-20251001). Forced tool_choice is supported.
AFD_MODEL = "claude-haiku-4-5"

# Fixed external endpoint (SSRF guard T-02-11: WFO codes come from the static map below,
# never from untrusted input; the host is a constant).
_NWS_API_BASE = "https://api.weather.gov"
# A descriptive User-Agent is REQUIRED by api.weather.gov (v3 NWS_UA shape, L77).
_USER_AGENT = "weatherquant/0.1 (kalshi daily-high paper-trading)"


# --- v3 pre-filter, ported VERBATIM (kalshi_weather_bot_v3.py L277-309) -------------
# These two keyword lists and afd_should_classify gate BEFORE any paid Anthropic call so
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
    """Keyword pre-filter — return ``(should_classify, reason)`` (ported VERBATIM from v3).

    Runs BEFORE any paid Anthropic call (D-13, budget). A routine opening phrase in the first
    300 chars short-circuits to ``(False, reason)`` (no API call); otherwise a signal keyword
    anywhere in the body yields ``(True, reason)``; absent both, ``(False, "no signal …")``.
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


# --- Anthropic forced tool-use (Pattern 7) -----------------------------------------
# A single tool whose input_schema IS the JSON shape. Forcing tool_choice to this tool
# makes the model return a guaranteed-shape dict (no decoding of a prose text body, D-13).
# Constraining to this schema bounds prompt-injection: injected AFD text cannot widen the
# output beyond {disagreement, direction, summary} (T-02-08, ASVS V5).
_AFD_TOOL = {
    "name": "record_afd_signal",
    "description": "Record the forecaster-disagreement signal from an NWS AFD excerpt.",
    "input_schema": {
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
}

# The no-signal result returned without any API call (pre-filter skip / unset key / degrade).
_NO_SIGNAL = {"disagreement": False, "direction": "", "summary": ""}


def classify_afd(
    text: str,
    wfo: str,
    client: object | None = None,
) -> dict:
    """Classify one AFD excerpt into a structured disagreement signal (ING-07, D-13).

    Order of operations:

    1. :func:`afd_should_classify` runs FIRST — a routine/no-signal text returns the
       no-signal dict (with the pre-filter ``reason``) and makes NO Anthropic call (budget).
    2. The ``ANTHROPIC_API_KEY`` is read from ``Settings`` (never the raw environment inline). If it
       is unset, a structured skip is logged and the no-signal dict is returned (D-11).
    3. Otherwise the Anthropic SDK is called with the single ``record_afd_signal`` tool and
       ``tool_choice`` forced to it — the returned ``tool_use`` ``input`` dict (the guaranteed
       JSON shape) is returned. The message text body is never decoded as JSON.

    Args:
        text: the raw AFD product text (untrusted free-text — T-02-08).
        wfo: the NWS Forecast Office code (from :data:`CITY_WFO`, a static map — T-02-11).
        client: optional pre-built Anthropic client (the unit test injects a mock so no real
            network/paid call is made); when ``None`` a real ``anthropic.Anthropic`` is built.

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
        api_key = get_settings().anthropic_api_key
        if not api_key:
            # Graceful degrade (D-11): the key is unset — log a structured skip, no call.
            logger.warning(
                "AFD classify skipped for WFO=%s: ANTHROPIC_API_KEY unset (degrade, D-11)",
                wfo,
            )
            return {**_NO_SIGNAL, "reason": "anthropic_api_key unset"}
        from anthropic import Anthropic

        client = Anthropic(api_key=api_key)

    message = client.messages.create(  # type: ignore[attr-defined]
        model=AFD_MODEL,
        max_tokens=256,
        tools=[_AFD_TOOL],
        tool_choice={"type": "tool", "name": "record_afd_signal"},
        messages=[
            {
                "role": "user",
                "content": f"WFO: {wfo}\n\nAFD excerpt:\n{text[:1500]}",
            }
        ],
    )

    # Forced tool_choice → the response carries exactly one tool_use block whose `input`
    # is the schema-shaped dict. The text body is never decoded as JSON (v3 fragility, D-13).
    for block in message.content:
        if getattr(block, "type", None) == "tool_use":
            result = dict(block.input)
            # Defensive: guarantee the three required keys exist even if the SDK changes.
            return {
                "disagreement": bool(result.get("disagreement", False)),
                "direction": result.get("direction", ""),
                "summary": result.get("summary", ""),
            }

    logger.warning("AFD classify for WFO=%s returned no tool_use block; degrading", wfo)
    return dict(_NO_SIGNAL)


async def fetch_afd_text(
    wfo: str,
    client: httpx.AsyncClient | None = None,
) -> str | None:
    """Fetch the latest AFD product text for ``wfo`` from api.weather.gov (Pattern 7).

    Ports v3's two-step fetch (kalshi_weather_bot_v3.py L878-894):
    ``/products/types/AFD/locations/{wfo}`` → latest id → ``/products/{id}`` → ``productText``.
    The host is a fixed constant and ``wfo`` comes from the static :data:`CITY_WFO` map, so
    there is no SSRF surface (T-02-11). Returns ``None`` on any error (graceful, D-11).
    """
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=15.0, headers={"User-Agent": _USER_AGENT})
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
        return prod_resp.json().get("productText", "")
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("AFD fetch failed for WFO=%s (%s)", wfo, exc)
        return None
    finally:
        if owns_client:
            await client.aclose()


def store_afd_signal(
    bind: object,
    city: str,
    target_date: date,
    result: dict,
    available_at: datetime | None = None,
    mode: str = "live",
) -> int:
    """Persist an AFD signal as an ``observations`` row via the audited writer (D-06/D-10/D-11).

    Routes through :func:`weatherquant.ingest.writer.insert_observation` with ``source='afd'``
    and ``detail`` jsonb = the tool result — never a hand-rolled Core insert, never a new table.

    POINT-IN-TIME INTEGRITY (CR-01, D-09). ``available_at`` MUST be the AFD product issuance
    time (the report time). The ``now(UTC)`` fallback is permitted ONLY in ``mode="live"`` —
    the instant the running system actually held the product. In ``mode="backfill"`` an
    explicit ``available_at`` (the recovered historical issuance time) is REQUIRED: stamping
    ``now()`` on a row reconstructed for a past date would make the datum appear available
    years late and silently corrupt Phase 6's no-look-ahead walk-forward (the CR-01 leak), so
    a missing ``available_at`` in backfill raises ``ValueError`` rather than defaulting to
    ``now()``.

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
            # CR-01: never stamp now() on a backfilled (historical) AFD row — that is the
            # wall-clock look-ahead leak this whole module exists to prevent (D-09). An
            # AvailabilityError (a CorrectnessError) so it fails LOUD if it ever surfaces
            # through the orchestrator, never a silent skip (WR-05). Still a ValueError.
            raise AvailabilityError(
                f"store_afd_signal in backfill requires an explicit available_at "
                f"(AFD issuance time) for city={city} target_date={target_date} — refusing to "
                f"stamp now() on a historical row (CR-01/D-09)"
            )
        # Live: now(UTC) is the instant the running system actually held the product (D-09).
        available_at = datetime.now(timezone.utc)
    return insert_observation(
        bind,
        city=city,
        target_date=target_date,
        source=SOURCE,
        detail=detail,
        available_at=available_at,
    )


__all__ = [
    "AFD_ROUTINE_PHRASES",
    "AFD_SIGNAL_KEYWORDS",
    "afd_should_classify",
    "classify_afd",
    "fetch_afd_text",
    "store_afd_signal",
    "CITY_WFO",
    "AFD_MODEL",
    "SOURCE",
]
