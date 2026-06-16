"""Open-Meteo ensemble members -> per-member Celsius->Kelvin (ING-05, D-05/D-07).

Pulls the GFS-seamless ensemble (31 members, ``temperature_2m_member00..member30``) from
``ensemble-api.open-meteo.com/v1/ensemble`` in ONE request (all members batched to stay
inside the free-tier budget — A3), buckets each member's hourly values into the LST
:func:`weatherquant.time.settlement_window`, takes each member's in-window MAX, and stores
each member as a SEPARATE ``forecasts`` row keyed by its member index.

THE V3 TRAP (Pitfall 3 / D-07). v3 requested the °F temperature unit and would have
written °F into a Kelvin column. This module takes the API-DEFAULT Celsius (no
``temperature_unit`` param) and converts each member value to Kelvin via the centralized
:func:`celsius_to_kelvin` converter. The tests assert the stored value lands in the
~280-320 K band, catching any accidental °F store (a °F daily-high would be ~50-110).

MEMBER AXIS (D-05). ``member00`` -> ``member=0`` (the control / primary), ``member01..member30``
-> ``member=1..30``. The provider-namespaced label is ``openmeteo`` for the control member
(0) and ``openmeteo:<member>`` for the perturbations (D-12) — never deduped with NOAA-decoded
``gfs``/``gefs`` or with ``wethr:*``.

LIVE-FORWARD ONLY (Open Question 2, RESOLVED). The free Open-Meteo ensemble exposes only
~3 ``past_days``; the deep historical calibration corpus comes exclusively from the NOAA
GRIB archive. This module fetches in LIVE mode and makes NO multi-year backfill attempt.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date, datetime, timezone
from typing import Any, cast

import httpx

from weatherquant.ingest.available_at import available_at
from weatherquant.ingest.sources._client import get_client, request_with_retry
from weatherquant.ingest.writer import Bind, insert_forecast
from weatherquant.registry import get_city
from weatherquant.time import SettlementWindow, settlement_window

logger = logging.getLogger(__name__)

# Fixed endpoint (SSRF guard T-02-15) — v3 OPEN_METEO_ENS_BASE (kalshi_weather_bot_v3.py L51).
OPEN_METEO_ENS_BASE = "https://ensemble-api.open-meteo.com/v1"

# GFS-seamless = 31 ensemble members (member00..member30), the NOAA-family ensemble in scope.
ENSEMBLE_MODEL = "gfs_seamless"
N_MEMBERS = 31  # member00 .. member30

# Provider-namespaced base label (D-12). Control member (0) -> "openmeteo"; perturbations
# (1..30) -> "openmeteo:<member>" — never deduped with NOAA gfs/gefs or wethr:* (D-12).
MODEL_BASE = "openmeteo"

_KELVIN_OFFSET = 273.15


def celsius_to_kelvin(temp_c: float) -> float:
    """The ONE °C->K conversion on the Open-Meteo path (Pitfall 3 / D-07).

    Open-Meteo returns the API-default Celsius; forecasts are Kelvin-only. Centralizing the
    conversion here means no caller inlines ``+ 273.15`` and the units boundary stays
    auditable — and the v3 °F-unit request is never reintroduced.
    """
    return temp_c + _KELVIN_OFFSET


def member_label(member: int) -> str:
    """Provider-namespaced model label for an Open-Meteo ensemble member (D-05/D-12).

    Member 0 (the control) -> ``"openmeteo"``; members 1..30 -> ``"openmeteo:<member>"``.
    Keeps the same underlying model from two providers as two distinct blend inputs.
    """
    return MODEL_BASE if member == 0 else f"{MODEL_BASE}:{member}"


# The base hourly variable requested from the /ensemble endpoint. The ensemble API expands
# this ONE requested variable into the full member set in the response: the bare
# ``temperature_2m`` series is the CONTROL (member 0), and ``temperature_2m_member01`` ..
# ``temperature_2m_member30`` are the 30 perturbations. There is NO ``temperature_2m_member00``
# variable — requesting it (or the explicit member00..member30 list) is a 400 "Data corrupted"
# error, so the request asks for the base variable and the response is demultiplexed below.
HOURLY_VAR = "temperature_2m"


def _member_var(member: int) -> str:
    """The Open-Meteo RESPONSE key for a member index in the ``/ensemble`` payload.

    Member 0 (the control) is the bare ``temperature_2m`` series; members 1..30 are
    ``temperature_2m_member01`` .. ``temperature_2m_member30``. (Open-Meteo has no
    ``temperature_2m_member00`` key — the control is unsuffixed.)
    """
    return HOURLY_VAR if member == 0 else f"{HOURLY_VAR}_member{member:02d}"


def _window_max_kelvin(
    times: Sequence[str], temps_c: Sequence[object], win: SettlementWindow
) -> float | None:
    """Take a member's in-window MAX hourly °C and convert to Kelvin (``None`` if no in-window)."""
    best_c: float | None = None
    for ts_s, temp in zip(times, temps_c):
        if temp is None:
            continue
        try:
            ts = datetime.fromisoformat(str(ts_s).replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts = ts.astimezone(timezone.utc)
        if not (win.start_utc <= ts < win.end_utc):
            continue  # half-open bucket — a wrong-LST-day hour cannot raise the member high
        try:
            value = float(cast(Any, temp))
        except (TypeError, ValueError):
            continue
        if best_c is None or value > best_c:
            best_c = value
    return celsius_to_kelvin(best_c) if best_c is not None else None


def parse_members(payload: dict[str, Any], win: SettlementWindow) -> dict[int, float]:
    """Parse the ``/ensemble`` payload into ``{member_index: in-window-max Kelvin}`` (D-05).

    Reads ``hourly.time`` once and each ``temperature_2m_memberNN`` series, bucketing each
    member through ``win`` and converting its in-window max from Celsius to Kelvin. Members
    with no in-window reading are omitted (graceful degrade, D-11). The ``memberNN`` suffix
    maps directly to the integer member axis (member00 -> 0 ... member30 -> 30).
    """
    hourly = payload.get("hourly", {}) or {}
    times = hourly.get("time", []) or []
    out: dict[int, float] = {}
    for member in range(N_MEMBERS):
        series = hourly.get(_member_var(member))
        if series is None:
            continue
        kelvin = _window_max_kelvin(times, series, win)
        if kelvin is not None:
            out[member] = kelvin
    return out


async def fetch_openmeteo_ensemble(
    city: str,
    target_date: date,
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[int, float]:
    """Fetch the Open-Meteo ensemble high per member for ``city``/``target_date`` (ING-05).

    Issues ONE request to ``/ensemble`` for ALL 31 members (batched — A3), taking the
    API-default Celsius (NEVER the °F temperature unit — Pitfall 3) and converting
    each member's in-window max to Kelvin. Returns ``{member_index: temp_kelvin}``. Live-
    forward only (Open Question 2 RESOLVED — no deep backfill). Pure fetch+parse so the unit
    test injects a ``MockTransport`` client and runs offline.

    Args:
        city: Kalshi city code (resolved via :func:`get_city`).
        target_date: the LST settlement (civil) date to forecast each member's high for.
        client: optional injected ``httpx.AsyncClient`` (the unit test passes a mock).
    """
    station = get_city(city)
    ds = target_date.isoformat()
    params = {
        "latitude": station.lat,
        "longitude": station.lon,
        # Request the BASE variable; the /ensemble endpoint expands it into all 31 members in
        # the response (control = bare temperature_2m, member01..member30 the perturbations).
        # Requesting the explicit member00..member30 list 400s ("Data corrupted"). ONE request
        # still returns every member (A3 — stay in budget).
        "hourly": HOURLY_VAR,
        "models": ENSEMBLE_MODEL,
        "start_date": ds,
        "end_date": ds,
        # NOTE: no temperature_unit -> API-default Celsius (Pitfall 3 / D-07).
    }
    owns_client = client is None
    client = client or get_client()
    try:
        resp = await request_with_retry(
            client, "GET", f"{OPEN_METEO_ENS_BASE}/ensemble", params=params
        )
        resp.raise_for_status()
        payload = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "Open-Meteo ensemble fetch failed for city=%s date=%s (%s) — degrading (D-11)",
            city,
            target_date,
            exc,
        )
        return {}
    finally:
        if owns_client:
            await client.aclose()

    win = settlement_window(station, target_date)
    return parse_members(payload, win)


def store_members(
    bind: Bind,
    city: str,
    target_date: date,
    members: dict[int, float],
    *,
    cycle: datetime | None = None,
    mode: str = "live",
) -> int:
    """Persist each ensemble member as a SEPARATE forecast row via the audited path (D-05).

    Each ``{member: temp_kelvin}`` entry routes through
    :func:`weatherquant.ingest.writer.insert_forecast` under the provider-namespaced label
    :func:`member_label` (``openmeteo`` / ``openmeteo:<member>``), the integer member axis,
    ``lead=0`` (the daily-high quantity), Kelvin payload, and ``available_at`` threaded with
    ``mode`` (WR-01 — not hardcoded ``"live"``, so the live/backfill seam is genuinely single,
    D-15). Open-Meteo is live-forward only (the free tier exposes ~3 past_days), so the
    orchestrator refuses to run it in backfill (WR-02). Returns the count of rows actually
    inserted (skips already-present identical rows, D-10).
    """
    station = get_city(city)
    cycle = cycle or datetime.now(timezone.utc)
    inserted = 0
    for member, temp_kelvin in sorted(members.items()):
        inserted += insert_forecast(
            bind,
            city=city,
            target_date=target_date,
            model=member_label(member),
            lead=0,
            member=member,
            temp_kelvin=temp_kelvin,
            cycle=cycle,
            station_lat=station.lat,
            station_lon=station.lon,
            grid_distance_m=0.0,
            # available_at honors the threaded mode (WR-01); the live branch ignores the label.
            available_at=available_at(cycle, member_label(member), mode),  # type: ignore[arg-type]
        )
    return inserted


__all__ = [
    "OPEN_METEO_ENS_BASE",
    "ENSEMBLE_MODEL",
    "MODEL_BASE",
    "N_MEMBERS",
    "celsius_to_kelvin",
    "member_label",
    "parse_members",
    "fetch_openmeteo_ensemble",
    "store_members",
]
