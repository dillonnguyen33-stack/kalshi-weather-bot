"""NWS gridpoint forecast -> settlement_window max, stored as label ``nws`` (ING-04).

Two-hop api.weather.gov flow (Pattern 4): GET ``/points/{lat},{lon}`` to resolve the
station's ``properties.forecastGridData`` URL, then GET that gridpoint URL and read
``properties.temperature`` — a ``{uom, values:[{validTime, value}, ...]}`` block where each
``validTime`` is an ISO instant plus an ISO-8601 *duration* (``2026-06-15T18:00:00+00:00/PT1H``)
covering an interval. The intervals are bucketed into the Phase-1
:func:`weatherquant.time.settlement_window` ``[start_utc, end_utc)`` and the in-window MAX is
the forecast high for ``target_date`` (the daily-high quantity Kalshi settles).

THE UNIT TRAP (Pitfall 3). NWS reports the unit explicitly in ``uom`` (a WMO unit code,
typically ``wmoUnit:degC``). This module NEVER assumes the unit — it reads ``uom`` and
converts to Kelvin via the centralized converter, raising on an unrecognized unit rather
than storing garbage (T-02-12). Forecasts are Kelvin-only (D-07); °F/°C never reach storage.

THE 403 TRAP (Pitfall 7). api.weather.gov 403s without a descriptive ``User-Agent`` — that
header is carried by the shared client (``_client.get_client``); this module also asserts it
on the request so a future client change cannot silently re-introduce the 403.

All rows route through 02-02's :func:`weatherquant.ingest.writer.insert_forecast` +
:func:`weatherquant.ingest.available_at.available_at` (live mode for the live fetch) under
the provider-namespaced label ``nws`` (D-12) — never deduped with NOAA-decoded models.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import httpx

from weatherquant.ingest.available_at import available_at
from weatherquant.ingest.sources._client import get_client, request_with_retry
from weatherquant.ingest.writer import insert_forecast
from weatherquant.registry import get_city
from weatherquant.time import SettlementWindow, settlement_window

logger = logging.getLogger(__name__)

# Fixed endpoint (SSRF guard T-02-15) — v3 NWS_API_BASE (kalshi_weather_bot_v3.py L52).
NWS_API_BASE = "https://api.weather.gov"

# Provider-namespaced model label (D-12) — never deduped with NOAA-decoded ``nbm``/``hrrr``.
MODEL = "nws"

# WMO unit codes api.weather.gov uses for ``properties.temperature.uom`` (Pitfall 3). The
# unit is read from the payload, NEVER assumed; an unrecognized code raises (T-02-12).
_KELVIN_OFFSET = 273.15


def _to_kelvin(value: float, uom: str) -> float:
    """Convert a temperature ``value`` in WMO unit ``uom`` to Kelvin (the centralized seam).

    Recognizes the WMO ``degC`` / ``degF`` / ``K`` codes (with or without the ``wmoUnit:``
    prefix). An unrecognized unit is a hard error (T-02-12 — never silently store the wrong
    unit into the Kelvin-only forecast path, D-07).
    """
    code = uom.split(":", 1)[1] if ":" in uom else uom
    code = code.strip().lower()
    if code in ("degc", "celsius", "c"):
        return value + _KELVIN_OFFSET
    if code in ("degf", "fahrenheit", "f"):
        return (value - 32.0) * 5.0 / 9.0 + _KELVIN_OFFSET
    if code in ("k", "kelvin"):
        return value
    raise ValueError(
        f"unrecognized NWS temperature unit {uom!r} — refusing to store an assumed unit "
        f"into the Kelvin-only forecast path (Pitfall 3 / D-07)"
    )


def _parse_valid_interval(valid_time: str) -> tuple[datetime, datetime]:
    """Parse an NWS ``validTime`` (``<ISO instant>/<ISO-8601 duration>``) to ``[start, end)``.

    NWS encodes each value's coverage as a start instant plus an ISO-8601 duration
    (``PnDTnHnMnS``). The interval is half-open ``[start, end)``; the duration is parsed
    explicitly (no third-party dep) supporting day/hour/minute/second components.
    """
    instant_s, _, duration_s = valid_time.partition("/")
    start = datetime.fromisoformat(instant_s.replace("Z", "+00:00"))
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    end = start + _parse_iso_duration(duration_s) if duration_s else start
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


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


def window_max_kelvin(temperature: dict, win: SettlementWindow) -> float | None:
    """Bucket ``properties.temperature`` intervals into ``win`` and return the in-window MAX (K).

    Each ``values`` entry contributes its (unit-converted) Kelvin value if its half-open
    ``[start, end)`` interval OVERLAPS the half-open settlement window — an interval that
    touches the window counts, an interval entirely before/after does not (so a hotter value
    on the wrong LST day cannot raise the high). Returns ``None`` when no interval overlaps.
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
    mode: str = "live",
) -> float | None:
    """Fetch the NWS gridpoint forecast high for ``city``/``target_date`` (ING-04, Pattern 4).

    Resolves the registry station lat/lon (:func:`get_city`), GETs ``/points/{lat},{lon}`` to
    obtain ``properties.forecastGridData``, GETs that gridpoint URL, buckets
    ``properties.temperature`` into the LST :func:`settlement_window`, converts the in-window
    max to Kelvin via the explicit ``uom`` (NEVER assumed — Pitfall 3), and returns the
    Kelvin high (``None`` if no forecast covers the window — graceful degrade, D-11). The
    required User-Agent (Pitfall 7) is carried by the shared client and asserted here. The
    caller persists via :func:`store_nws_forecast`; this function is pure fetch+parse so the
    unit test can inject a ``MockTransport`` client and run offline.

    Args:
        city: Kalshi city code (resolved via :func:`get_city`; unknown raises KeyError).
        target_date: the LST settlement (civil) date to forecast the daily high for.
        client: optional injected ``httpx.AsyncClient`` (the unit test passes a mock).
        mode: unused here (kept for signature symmetry with the writer-routing helper).
    """
    station = get_city(city)
    owns_client = client is None
    client = client or get_client()
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
    finally:
        if owns_client:
            await client.aclose()

    win = settlement_window(station, target_date)
    return window_max_kelvin(temperature, win)


def store_nws_forecast(
    bind: object,
    city: str,
    target_date: date,
    temp_kelvin: float,
    *,
    cycle: datetime | None = None,
    mode: str = "live",
) -> int:
    """Persist one NWS forecast row via the SINGLE audited writer path (D-10/D-11).

    Routes through :func:`weatherquant.ingest.writer.insert_forecast` under label ``nws``
    (D-12), ``member=0`` (deterministic), ``lead=0`` (the daily-high quantity for the target
    day), Kelvin payload, and ``available_at`` from the helper threaded with ``mode``. The
    station snap fields are the registry station's own lat/lon with ``grid_distance_m=0.0``
    (NWS already returns the gridpoint for the station, so there is no separate haversine snap
    here).

    ``mode`` is threaded into :func:`available_at` rather than hardcoded ``"live"`` so the
    live/backfill seam is genuinely SINGLE (D-15, WR-01). NWS is a live-forward source whose
    only historical archive is the NOAA GRIB corpus, so the orchestrator refuses to run it in
    backfill (WR-02); this signature keeps the seam honest if it is ever called with a mode.

    Returns:
        ``1`` if a row was inserted, ``0`` if an identical row already existed (skip).
    """
    station = get_city(city)
    cycle = cycle or datetime.now(timezone.utc)
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
    "NWS_API_BASE",
    "MODEL",
    "fetch_nws_forecast",
    "store_nws_forecast",
    "window_max_kelvin",
]
