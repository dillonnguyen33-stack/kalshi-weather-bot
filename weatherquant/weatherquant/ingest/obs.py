"""Ground-truth daily-high observations (ING-03) — bucket ASOS/METAR through the LST window.

Produces the verifying truth Phase 3/6 use. The correctness core is the bucket: daily-high =
``max(tmpf)`` over the half-open ``settlement_window`` (the one LST authority), never a
hand-rolled UTC day (D-16/D-01/D-02). The only °F seam is :func:`celsius_to_fahrenheit`.
Writes route through the single audited :func:`insert_observation`; the obs ``available_at``
is the feed's report time, not ``now()`` (D-09/D-10/D-11). A CLI-vs-ASOS disagreement is
flagged but the ASOS label is still stored, never overwritten (D-16; see docs/DECISIONS.md).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, UTC
from typing import Any, cast

import httpx

from weatherquant.ingest.sources._client import KELVIN_OFFSET, managed_client
from weatherquant.ingest.writer import Bind, insert_observation
from weatherquant.registry import get_city
from weatherquant.time import SettlementWindow, coerce_utc, parse_utc, settlement_window

logger = logging.getLogger(__name__)

# Fixed external endpoints (SSRF guard T-02-11: never built from untrusted input).
_IEM_ASOS_CGI = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
# v3 AWC_METAR_BASE — the recent-~14h *live fallback* only (kalshi_weather_bot_v3.py L53).
_AWC_METAR_BASE = "https://aviationweather.gov/api/data/metar"
# AWC lookback: the feed returns only the most recent ~N hours of METARs, so it can serve a
# live/today window only — a window ending older than this predates the feed (backfill, skip).
_AWC_FALLBACK_HOURS = 14
# v3 NWS_UA (L77) — a descriptive User-Agent is required/courteous for these feeds.
_USER_AGENT = "weatherquant/0.1 (kalshi daily-high paper-trading)"

# CLI-vs-ASOS agreement tolerance in °F. The CLI report rounds to whole degrees and the
# ASOS max may differ by sub-degree sampling, so a 1.5°F band is a reasonable
# data-quality threshold before flagging a genuine disagreement (D-16).
_CLI_DISAGREEMENT_TOLERANCE_F = 1.5

SOURCE = "asos"


def celsius_to_fahrenheit(temp_c: float) -> float:
    """The ONE °C→°F conversion on the obs path, keeping the units boundary auditable (D-07)."""
    return temp_c * 9.0 / 5.0 + 32.0


@dataclass(frozen=True)
class DailyHigh:
    """The bucketed daily-high label plus its provenance (ING-03).

    Attributes:
        window_start/window_end: the half-open ``[start, end)`` UTC settlement window.
        report_time: timestamp of the max reading — the obs ``available_at`` source (D-09).
        cli_disagreement: ``True`` when a CLI max disagreed beyond tolerance; label still
            produced, never overwritten (D-16).
    """

    daily_high_f: float | None
    obs_count: int
    window_start: datetime
    window_end: datetime
    report_time: datetime | None = None
    cli_disagreement: bool = False
    cli_max_f: float | None = field(default=None)


def _coerce(reading: object) -> tuple[datetime, float] | None:
    """Extract a well-formed ``(ts_utc, tmpf)`` from one untrusted feed row, else ``None`` (T-02-07).

    Accepts a ``(ts, tmpf)`` pair or a mapping with ``ts``/``ts_utc`` + ``tmpf``/``temp_f``;
    naive timestamps are assumed UTC.
    """
    from collections.abc import Mapping

    ts_raw: object
    tmpf_raw: object
    if isinstance(reading, Mapping):
        ts_raw = reading.get("ts_utc", reading.get("ts"))
        tmpf_raw = reading.get("tmpf", reading.get("temp_f"))
    elif isinstance(reading, Sequence) and not isinstance(reading, (str, bytes)):
        if len(reading) < 2:
            return None
        ts_raw, tmpf_raw = reading[0], reading[1]
    else:
        return None

    ts = _coerce_ts(ts_raw)
    if ts is None:
        return None
    try:
        if tmpf_raw is None:
            return None
        # Untrusted parsed value (CSV/JSON edge); the except below catches a bad type.
        tmpf = float(cast(Any, tmpf_raw))
    except (TypeError, ValueError):
        return None
    return ts, tmpf


def _coerce_ts(ts_raw: object) -> datetime | None:
    """Coerce a feed timestamp to a tz-aware UTC ``datetime`` (skip if uncoercible)."""
    if isinstance(ts_raw, datetime):
        ts = ts_raw
    elif isinstance(ts_raw, str):
        try:
            ts = parse_utc(ts_raw)
        except ValueError:
            return None
    else:
        return None
    return coerce_utc(ts).astimezone(UTC)


def daily_high(
    rows: Iterable[object],
    city: str,
    target_date: date,
    cli_max_f: float | None = None,
) -> DailyHigh:
    """Bucket ``rows`` to the half-open LST settlement window and return the daily-high label (ING-03).

    A reading at exactly ``end_utc`` is excluded and one just outside cannot raise the high;
    malformed rows are skipped (T-02-07). A CLI disagreement beyond tolerance is flagged but
    the ASOS label is still produced (D-16; see docs/DECISIONS.md).

    Args:
        rows: untrusted feed readings — ``(ts_utc, tmpf)`` pairs or mappings; temps already °F.
        cli_max_f: optional NWS CLI oracle max for the data-quality cross-check (D-16).
    """
    win: SettlementWindow = settlement_window(get_city(city), target_date)

    best_f: float | None = None
    best_ts: datetime | None = None
    count = 0
    for raw in rows:
        coerced = _coerce(raw)
        if coerced is None:
            continue
        ts, tmpf = coerced
        # Half-open [start, end): end_utc is EXCLUSIVE (no double-count, no flat window).
        if not win.contains(ts):
            continue
        count += 1
        if best_f is None or tmpf > best_f:
            best_f = tmpf
            best_ts = ts

    disagreement = False
    if cli_max_f is not None and best_f is not None:
        if abs(best_f - cli_max_f) > _CLI_DISAGREEMENT_TOLERANCE_F:
            disagreement = True
            # D-16: flag + log the data-quality event; do NOT overwrite the ASOS label.
            logger.warning(
                "obs CLI disagreement city=%s date=%s asos_max=%.1fF cli_max=%.1fF "
                "(>%.1fF tolerance) — storing ASOS label, flagging event",
                city,
                target_date,
                best_f,
                cli_max_f,
                _CLI_DISAGREEMENT_TOLERANCE_F,
            )

    return DailyHigh(
        daily_high_f=best_f,
        obs_count=count,
        window_start=win.start_utc,
        window_end=win.end_utc,
        report_time=best_ts,
        cli_disagreement=disagreement,
        cli_max_f=cli_max_f,
    )


def daily_high_from_obs(
    city: str,
    target_date: date,
    readings: Iterable[object],
    cli_max_f: float | None = None,
) -> DailyHigh:
    """Arg-reordered alias of :func:`daily_high` matching the 02-01 RED-stub ``(city, target_date, readings)``."""
    return daily_high(readings, city, target_date, cli_max_f=cli_max_f)


def _parse_iem_csv(body: str) -> list[tuple[datetime, float]]:
    """Parse the IEM ``onlycomma`` CSV (``station,valid,tmpf``, tz=UTC) into ``(ts_utc, tmpf)`` rows.

    Missing values (``M``) are skipped (T-02-07).
    """
    rows: list[tuple[datetime, float]] = []
    for line in body.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("station"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        _station, valid, tmpf_s = parts[0], parts[1], parts[2]
        if tmpf_s.strip().upper() in ("M", "", "T"):
            continue
        try:
            # IEM emits ``YYYY-MM-DD HH:MM`` UTC; parse_utc (fromisoformat) handles the space
            # separator and stamps the naive instant as UTC — same result, one parse seam.
            ts = parse_utc(valid.strip())
            tmpf = float(tmpf_s)
        except ValueError:
            continue
        rows.append((ts, tmpf))
    return rows


async def fetch_asos_obs(
    city: str,
    win: SettlementWindow,
    client: httpx.AsyncClient | None = None,
) -> list[tuple[datetime, float]]:
    """Fetch sub-daily ASOS/METAR ``(ts_utc, tmpf)`` rows for ``win`` (ING-03, Pattern 3).

    Primary feed is the IEM ASOS CGI keyed by the registry ``cli_station``; on failure it
    degrades to the aviationweather.gov METAR endpoint (°C → °F). Fixed endpoints (SSRF guard
    T-02-11). Returns raw rows; :func:`daily_high` does the bucket (D-11).
    """
    station = get_city(city).cli_station
    # IEM strips the leading 'K' from the 4-letter ICAO for its 3-letter network ids.
    iem_station = station[1:] if station.startswith("K") and len(station) == 4 else station

    async with managed_client(client) as client:
        try:
            resp = await client.get(
                _IEM_ASOS_CGI,
                params={
                    "station": iem_station,
                    "data": "tmpf",
                    "sts": win.start_utc.strftime("%Y-%m-%dT%H:%MZ"),
                    "ets": win.end_utc.strftime("%Y-%m-%dT%H:%MZ"),
                    "tz": "UTC",
                    "report_type": ["3", "4"],  # routine (3) + special (4) METAR
                    "format": "onlycomma",
                    "missing": "M",
                },
            )
            resp.raise_for_status()
            # A successful response is authoritative — empty rows mean "no obs for this window"
            # (a real answer), NOT a failure. Only a genuine fetch error falls back to AWC.
            return _parse_iem_csv(resp.text)
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("IEM ASOS fetch failed for %s (%s); trying AWC fallback", city, exc)
            return await _fetch_awc_fallback(station, client, win)


async def _fetch_awc_fallback(
    station: str, client: httpx.AsyncClient, win: SettlementWindow
) -> list[tuple[datetime, float]]:
    """aviationweather.gov METAR fallback — LIVE/recent windows only (°C → °F).

    The feed returns only the most recent ~``_AWC_FALLBACK_HOURS`` of METARs, so it can cover a
    today/live window. A window ending older than that lookback (a historical backfill) is
    skipped — fetching the recent feed would stamp today's obs onto a past date. Returned obs
    are filtered to ``win`` so nothing outside the target window leaks in.
    """
    if win.end_utc <= datetime.now(UTC) - timedelta(hours=_AWC_FALLBACK_HOURS):
        logger.info(
            "AWC fallback skipped for %s: window ending %s predates the ~%dh recent feed "
            "(backfill — no wrong-date obs)",
            station,
            win.end_utc.isoformat(),
            _AWC_FALLBACK_HOURS,
        )
        return []
    try:
        resp = await client.get(
            _AWC_METAR_BASE,
            params={"ids": station, "format": "json", "hours": _AWC_FALLBACK_HOURS},
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
        data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("AWC METAR fallback also failed for %s (%s)", station, exc)
        return []

    rows: list[tuple[datetime, float]] = []
    for ob in data or []:
        temp_c = ob.get("temp")
        ts = _coerce_ts(ob.get("reportTime") or ob.get("obsTime"))
        if temp_c is None or ts is None or not win.contains(ts):
            continue  # filter to the target window — no wrong-date leak
        try:
            rows.append((ts, celsius_to_fahrenheit(float(temp_c))))
        except (TypeError, ValueError):
            continue
    return rows


async def asos_lead0_kelvin(
    city: str,
    target_date: date,
    instant: datetime,
    *,
    client: httpx.AsyncClient | None = None,
) -> float | None:
    """ASOS reading nearest ``instant`` in Kelvin for the lead-0 sanity probe (D-04), else ``None``.

    Reuses the same settlement-window ASOS fetch the daily-high uses, returns the in-window
    reading closest to ``instant`` converted °F → Kelvin (the obs-path °F→K seam). ``None`` when
    no obs cover the window so the caller can skip the probe rather than fabricate a comparison.
    """
    win = settlement_window(get_city(city), target_date)
    rows = await fetch_asos_obs(city, win, client=client)
    nearest_f: float | None = None
    nearest_gap: float | None = None
    for raw in rows:
        coerced = _coerce(raw)
        if coerced is None:
            continue
        ts, tmpf = coerced
        if not win.contains(ts):
            continue
        gap = abs((ts - instant).total_seconds())
        if nearest_gap is None or gap < nearest_gap:
            nearest_gap = gap
            nearest_f = tmpf
    if nearest_f is None:
        return None
    return (nearest_f - 32.0) * 5.0 / 9.0 + KELVIN_OFFSET


def store_daily_high(
    bind: Bind,
    city: str,
    target_date: date,
    result: DailyHigh,
) -> int:
    """Persist a daily-high label via the SINGLE audited writer path (D-10/D-11).

    ``available_at`` is the obs report time (the max reading), not ``now()`` (D-09), falling
    back to ``window_end`` only for an empty label; the CLI flag is kept in ``detail`` (D-16).

    Returns:
        ``1`` if a row was inserted, ``0`` if an identical row already existed (skip).
    """
    available_at = result.report_time or result.window_end
    detail = {
        "cli_disagreement": result.cli_disagreement,
        "cli_max_f": result.cli_max_f,
    }
    return insert_observation(
        bind,
        city=city,
        target_date=target_date,
        source=SOURCE,
        daily_high_f=result.daily_high_f,
        window_start=result.window_start,
        window_end=result.window_end,
        obs_count=result.obs_count,
        detail=detail,
        available_at=available_at,
    )


__all__ = [
    "SOURCE",
    "DailyHigh",
    "asos_lead0_kelvin",
    "celsius_to_fahrenheit",
    "daily_high",
    "daily_high_from_obs",
    "fetch_asos_obs",
    "store_daily_high",
]
