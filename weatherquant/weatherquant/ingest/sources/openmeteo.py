"""Open-Meteo ensemble members -> per-member Celsius->Kelvin (ING-05, D-05/D-07).

Pulls the GFS-seamless ensemble (31 members) in ONE request (batched for budget — A3), buckets
each member into the LST window, takes its in-window max, and stores each as a separate
``forecasts`` row keyed by member index. Takes the API-default Celsius and converts via
:func:`celsius_to_kelvin` (never the v3 °F trap, Pitfall 3 / D-07). Members map member00→0 /
member01..member30→1..30 under namespaced labels ``openmeteo`` / ``openmeteo:<member>`` (D-05/
D-12). Live-forward only: the free tier exposes ~3 past_days, so no backfill (see
docs/DECISIONS.md).
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
    """The ONE °C->K conversion on the Open-Meteo path, keeping the units boundary auditable (D-07)."""
    return temp_c + _KELVIN_OFFSET


def member_label(member: int) -> str:
    """Namespaced label for a member: 0 → ``"openmeteo"``, 1..30 → ``"openmeteo:<member>"`` (D-05/D-12)."""
    return MODEL_BASE if member == 0 else f"{MODEL_BASE}:{member}"


# The base variable requested from /ensemble; the API expands it into all members in the
# response (bare ``temperature_2m`` = control, ``_member01..30`` = perturbations). There is no
# ``_member00`` and requesting the explicit list 400s, so the request asks for the base and the
# response is demultiplexed below.
HOURLY_VAR = "temperature_2m"


def _member_var(member: int) -> str:
    """Response key for a member: 0 → bare ``temperature_2m``, 1..30 → ``temperature_2m_memberNN``."""
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

    Members with no in-window reading are omitted (degrade, D-11).
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

    One request for all 31 members (batched — A3), API-default Celsius → Kelvin (never °F,
    Pitfall 3). Returns ``{member_index: temp_kelvin}``. Live-forward only; pure fetch+parse so
    the unit test injects a ``MockTransport`` client offline.

    Args:
        client: optional injected ``httpx.AsyncClient`` (the unit test passes a mock).
    """
    station = get_city(city)
    ds = target_date.isoformat()
    params = {
        "latitude": station.lat,
        "longitude": station.lon,
        # Base variable; /ensemble expands it into all 31 members (see HOURLY_VAR), one request
        # (A3 — stay in budget).
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

    Each entry stores under :func:`member_label`, the member axis, ``lead=0``, Kelvin payload,
    and ``available_at`` threaded with ``mode`` (WR-01, so the seam stays single — D-15;
    orchestrator refuses backfill, WR-02). Returns rows inserted (skips identical, D-10).
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
