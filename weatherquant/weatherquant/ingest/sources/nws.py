"""NWS gridpoint forecast -> settlement_window max, stored as label ``nws`` (ING-04).

Two-hop api.weather.gov flow (Pattern 4): ``/points/{lat},{lon}`` → the gridpoint URL →
``properties.temperature``, whose ``{uom, values:[{validTime, value}]}`` intervals (each an
ISO instant + ISO-8601 duration) are bucketed into the half-open ``settlement_window``; the
in-window max is the forecast high. The unit is read from ``uom`` and converted to Kelvin,
raising on an unrecognized unit (Pitfall 3 / T-02-12 / D-07); a descriptive ``User-Agent`` is
asserted to avoid the 403 trap (Pitfall 7). Rows route through the single audited writer under
the namespaced label ``nws`` (D-12; see docs/DECISIONS.md).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, UTC
from typing import Any

import httpx

from weatherquant.ingest.available_at import available_at
from weatherquant.ingest.errors import UnitError
from weatherquant.ingest.sources._client import (
    KELVIN_OFFSET,
    managed_client,
    request_with_retry,
)
from weatherquant.ingest.writer import Bind, insert_forecast
from weatherquant.registry import get_city
from weatherquant.time import SettlementWindow, parse_utc, settlement_window

logger = logging.getLogger(__name__)

# Fixed endpoint (SSRF guard T-02-15) — v3 NWS_API_BASE (kalshi_weather_bot_v3.py L52).
NWS_API_BASE = "https://api.weather.gov"

# Provider-namespaced model label (D-12) — never deduped with NOAA-decoded ``nbm``/``hrrr``.
MODEL = "nws"

# WMO unit codes api.weather.gov uses for ``properties.temperature.uom`` (Pitfall 3). The
# unit is read from the payload, NEVER assumed; an unrecognized code raises (T-02-12).


def _to_kelvin(value: float, uom: str) -> float:
    """Convert a temperature ``value`` in WMO unit ``uom`` to Kelvin, the centralized seam (D-07).

    Recognizes ``degC``/``degF``/``K`` (with or without the ``wmoUnit:`` prefix); an
    unrecognized unit is a hard error (T-02-12).
    """
    code = uom.split(":", 1)[1] if ":" in uom else uom
    code = code.strip().lower()
    if code in ("degc", "celsius", "c"):
        return value + KELVIN_OFFSET
    if code in ("degf", "fahrenheit", "f"):
        return (value - 32.0) * 5.0 / 9.0 + KELVIN_OFFSET
    if code in ("k", "kelvin"):
        return value
    # An unrecognized unit is a correctness alarm (UnitError): storing an assumed unit into the
    # Kelvin-only path would corrupt the ledger, so it must fail LOUD and NOT degrade to a
    # silent skip (WR-05). Still a ValueError for any caller catching that.
    raise UnitError(
        f"unrecognized NWS temperature unit {uom!r} — refusing to store an assumed unit "
        f"into the Kelvin-only forecast path (Pitfall 3 / D-07)"
    )


def _parse_valid_interval(valid_time: str) -> tuple[datetime, datetime]:
    """Parse an NWS ``validTime`` (``<ISO instant>/<ISO-8601 duration>``) to half-open ``[start, end)``."""
    instant_s, _, duration_s = valid_time.partition("/")
    start = parse_utc(instant_s)
    end = start + _parse_iso_duration(duration_s) if duration_s else start
    return start.astimezone(UTC), end.astimezone(UTC)


def _parse_iso_duration(duration: str) -> "datetime.timedelta":  # type: ignore[name-defined]
    """Parse an ISO-8601 duration (``PnDTnHnMnS``) into a ``timedelta`` (days/h/m/s only)."""
    from datetime import timedelta

    if not duration.startswith("P"):
        raise ValueError(f"not an ISO-8601 duration: {duration!r}")
    days = hours = minutes = seconds = 0
    body = duration[1:]
    date_part, _, time_part = body.partition("T")
    if date_part.endswith("D"):
        days = int(date_part[:-1])
    elif date_part:
        raise ValueError(f"unsupported ISO duration date part: {duration!r}")
    num = ""
    for ch in time_part:
        if ch.isdigit():
            num += ch
        elif ch == "H":
            hours = int(num)
            num = ""
        elif ch == "M":
            minutes = int(num)
            num = ""
        elif ch == "S":
            seconds = int(num)
            num = ""
        else:
            raise ValueError(f"unsupported ISO duration time part: {duration!r}")
    return timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)


def window_max_kelvin(temperature: dict[str, Any], win: SettlementWindow) -> float | None:
    """Bucket ``properties.temperature`` intervals into ``win`` and return the in-window MAX K.

    A ``values`` entry counts if its half-open interval overlaps the window, so a hotter value
    on the wrong LST day cannot raise the high; ``None`` if none overlap.
    """
    uom = temperature.get("uom", "")
    best: float | None = None
    for entry in temperature.get("values", []) or []:
        value = entry.get("value")
        valid_time = entry.get("validTime")
        if value is None or not valid_time:
            continue  # skip malformed entries (T-02-12), never store garbage
        i_start, i_end = _parse_valid_interval(valid_time)
        # Half-open interval overlap with the half-open window [start, end).
        if i_end <= win.start_utc or i_start >= win.end_utc:
            continue
        kelvin = _to_kelvin(float(value), uom)
        if best is None or kelvin > best:
            best = kelvin
    return best


async def fetch_nws_forecast(
    city: str,
    target_date: date,
    *,
    client: httpx.AsyncClient | None = None,
) -> float | None:
    """Fetch the NWS gridpoint forecast high for ``city``/``target_date`` (ING-04, Pattern 4).

    Buckets ``properties.temperature`` into the LST window and returns the Kelvin in-window max
    (``None`` if no forecast covers it — degrade, D-11); the User-Agent is asserted (Pitfall 7).
    Pure fetch+parse so the unit test injects a ``MockTransport`` client offline.

    Args:
        client: optional injected ``httpx.AsyncClient`` (the unit test passes a mock).
    """
    station = get_city(city)
    async with managed_client(client) as client:
        # Defense in depth (Pitfall 7): the shared client sets the User-Agent default, but assert
        # it so a future client refactor cannot silently re-introduce the api.weather.gov 403.
        assert client.headers.get("User-Agent"), "NWS requests require a User-Agent (Pitfall 7)"
        try:
            points_url = f"{NWS_API_BASE}/points/{station.lat},{station.lon}"
            points_resp = await request_with_retry(
                client, "GET", points_url, headers={"Accept": "application/geo+json"}
            )
            points_resp.raise_for_status()
            grid_url = points_resp.json()["properties"]["forecastGridData"]

            grid_resp = await request_with_retry(
                client, "GET", grid_url, headers={"Accept": "application/geo+json"}
            )
            grid_resp.raise_for_status()
            temperature = grid_resp.json()["properties"]["temperature"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.warning(
                "NWS gridpoint fetch failed for city=%s date=%s (%s) — degrading (D-11)",
                city,
                target_date,
                exc,
            )
            return None

    win = settlement_window(station, target_date)
    return window_max_kelvin(temperature, win)


def store_nws_forecast(
    bind: Bind,
    city: str,
    target_date: date,
    temp_kelvin: float,
    *,
    cycle: datetime | None = None,
    mode: str = "live",
) -> int:
    """Persist one NWS forecast row via the SINGLE audited writer path (D-10/D-11).

    Stores under label ``nws`` (D-12), ``member=0``, ``lead=0``, Kelvin payload, station snap =
    the registry lat/lon with ``grid_distance_m=0.0`` (NWS already returns the gridpoint).
    ``mode`` is threaded into :func:`available_at`, not hardcoded, so the live/backfill seam
    stays single (D-15/WR-01; orchestrator refuses backfill, WR-02).

    Returns:
        ``1`` if a row was inserted, ``0`` if an identical row already existed (skip).
    """
    station = get_city(city)
    cycle = cycle or datetime.now(UTC)
    return insert_forecast(
        bind,
        city=city,
        target_date=target_date,
        model=MODEL,
        lead=0,
        member=0,
        temp_kelvin=temp_kelvin,
        cycle=cycle,
        station_lat=station.lat,
        station_lon=station.lon,
        grid_distance_m=0.0,
        # available_at honors the threaded mode (WR-01). The live branch ignores the model
        # label, so the provider-namespaced "nws" never needs a PUBLISH_LATENCY entry (D-09).
        available_at=available_at(cycle, MODEL, mode),  # type: ignore[arg-type]
    )


__all__ = [
    "MODEL",
    "NWS_API_BASE",
    "fetch_nws_forecast",
    "store_nws_forecast",
    "window_max_kelvin",
]
