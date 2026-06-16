"""Ground-truth daily-high observations (ING-03) — bucket ASOS/METAR through the LST window.

This module produces the *verifying truth* Phase 3 calibrates against and Phase 6 scores.
The single correctness core is the bucket: the daily-high label is ``max(tmpf)`` over the
**half-open** Phase-1 ``settlement_window`` ``[start_utc, end_utc)`` — NEVER a hand-rolled
UTC day (D-16). This fixes v3's bug, which took the max over a flat ~14h aviationweather.gov
window (``fetch_asos_high_fallback`` at kalshi_weather_bot_v3.py L779-796): a flat window
both straddles the wrong LST day and silently clips the true intra-day peak.

The civil-time→UTC conversion is delegated entirely to :func:`weatherquant.time.settlement_window`
(the ONE LST authority — D-01/D-02); this module performs no DST-aware tz math and imports no
runtime DST tooling. The only °F seam in the obs path is centralized here in
:func:`celsius_to_fahrenheit` — the v3 inline ``*9/5+32`` (L793) lives in exactly one place.

Writes route through 02-02's :func:`weatherquant.ingest.writer.insert_observation` — the single
audited insert + skip-before-insert idempotency + ``rowcount==1`` path (D-10/D-11). There is no
hand-rolled Core insert here, so the observations table has exactly one write contract.

The obs row's ``available_at`` is the obs **report time** (the feed's own timestamp), NOT
``now()`` and NOT the window edge (D-09) — conflating those clocks would corrupt Phase 6's
no-look-ahead walk-forward.

A disagreement between the ASOS-derived max and the NWS CLI oracle (Phase-1 D-04) is a flagged
data-quality EVENT: the flag is logged and surfaced, but the ASOS label is STILL produced and
stored — never silently overwritten (D-16).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, cast

import httpx

from weatherquant.ingest.writer import Bind, insert_observation
from weatherquant.registry import get_city
from weatherquant.time import SettlementWindow, settlement_window

logger = logging.getLogger(__name__)

# Fixed external endpoints (SSRF guard T-02-11: never built from untrusted input).
_IEM_ASOS_CGI = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
# v3 AWC_METAR_BASE — the recent-~14h *live fallback* only (kalshi_weather_bot_v3.py L53).
_AWC_METAR_BASE = "https://aviationweather.gov/api/data/metar"
# v3 NWS_UA (L77) — a descriptive User-Agent is required/courteous for these feeds.
_USER_AGENT = "weatherquant/0.1 (kalshi daily-high paper-trading)"

# CLI-vs-ASOS agreement tolerance in °F. The CLI report rounds to whole degrees and the
# ASOS max may differ by sub-degree sampling, so a 1.5°F band is a reasonable
# data-quality threshold before flagging a genuine disagreement (D-16).
_CLI_DISAGREEMENT_TOLERANCE_F = 1.5

SOURCE = "asos"


def celsius_to_fahrenheit(temp_c: float) -> float:
    """The ONE °C→°F conversion on the obs path (centralized v3 L793 ``*9/5+32``).

    The obs payload is stored in °F (D-07); the only K↔F seam in the system lives between
    phases 2 and 3. Keeping this conversion in a single named function means no other module
    inlines ``* 9 / 5 + 32`` and the units boundary stays auditable.
    """
    return temp_c * 9.0 / 5.0 + 32.0


@dataclass(frozen=True)
class DailyHigh:
    """The bucketed daily-high label plus its provenance (ING-03).

    Attributes:
        daily_high_f: ``max(tmpf)`` over the in-window readings, in °F (``None`` if no
            reading fell inside the half-open window).
        obs_count: the number of readings that fell inside ``[window_start, window_end)``.
        window_start: the inclusive UTC start of the LST settlement window.
        window_end: the EXCLUSIVE UTC end of the LST settlement window (half-open).
        report_time: the timestamp of the reading that produced the max — the obs
            ``available_at`` is stamped from this (the report time, D-09), never ``now()``.
        cli_disagreement: ``True`` when a supplied CLI max disagreed with the ASOS max
            beyond tolerance — the label is still produced, never overwritten (D-16).
        cli_max_f: the CLI oracle max that was compared against (``None`` if not supplied).
    """

    daily_high_f: float | None
    obs_count: int
    window_start: datetime
    window_end: datetime
    report_time: datetime | None = None
    cli_disagreement: bool = False
    cli_max_f: float | None = field(default=None)


def _coerce(reading: object) -> tuple[datetime, float] | None:
    """Extract a well-formed ``(ts_utc, tmpf)`` from one untrusted feed row (T-02-07).

    Accepts a ``(ts, tmpf)`` pair or a mapping with ``ts``/``ts_utc`` and ``tmpf``/``temp_f``
    keys. Returns ``None`` (skip — never store garbage) for any malformed/uncoercible row.
    Naive timestamps are assumed UTC (the feeds are queried with ``tz=UTC``).
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
            ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def daily_high(
    rows: Iterable[object],
    city: str,
    target_date: date,
    cli_max_f: float | None = None,
) -> DailyHigh:
    """Bucket ``rows`` to the LST settlement window and return the daily-high label (ING-03).

    The window comes from :func:`settlement_window` (the ONLY LST authority); rows are kept
    only when ``win.start_utc <= ts < win.end_utc`` — the half-open boundary means a reading
    at exactly ``end_utc`` is EXCLUDED, and a hotter reading just OUTSIDE the window cannot
    raise the daily high (proving correct bucketing, not a flat window). Malformed rows are
    skipped (T-02-07). When ``cli_max_f`` is supplied and disagrees beyond tolerance, the
    returned ``DailyHigh`` carries ``cli_disagreement=True`` (logged) but the ASOS label is
    still produced — never overwritten (D-16).

    Args:
        rows: untrusted feed readings — ``(ts_utc, tmpf)`` pairs or mappings with
            ``ts``/``ts_utc`` + ``tmpf``/``temp_f``. Temperatures are already °F.
        city: Kalshi city code (resolved via :func:`get_city`).
        target_date: the LST settlement (civil) date being labeled.
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
        if not (win.start_utc <= ts < win.end_utc):
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
    """Keyword-friendly alias of :func:`daily_high` matching the 02-01 RED-stub contract.

    The Wave-0 stub (``tests/test_obs_daily_high.py``) imports this name and asserts the
    result exposes ``daily_high_f`` and ``obs_count``. One implementation — this just
    re-orders the arguments to the stub's ``(city, target_date, readings)`` shape.
    """
    return daily_high(readings, city, target_date, cli_max_f=cli_max_f)


def _parse_iem_csv(body: str) -> list[tuple[datetime, float]]:
    """Parse the IEM ``asos.py`` ``format=onlycomma`` CSV into ``(ts_utc, tmpf)`` rows.

    Columns: ``station,valid,tmpf`` (``valid`` is ``YYYY-MM-DD HH:MM`` in the requested
    ``tz=UTC``). Missing values are ``M`` — those rows are skipped (T-02-07).
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
            ts = datetime.strptime(valid.strip(), "%Y-%m-%d %H:%M").replace(
                tzinfo=timezone.utc
            )
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

    Primary feed: the IEM ASOS request CGI (``asos.py``) — ``data=tmpf``, ``sts``/``ets``
    set to ``win.start_utc``/``win.end_utc`` as UTC ISO, ``tz=UTC``, routine+special report
    types, ``format=onlycomma``. The station is the registry CLI station (``get_city(city).
    cli_station``), NOT v3's CITY_COORDS ICAO. On any IEM failure the live aviationweather.gov
    METAR endpoint (v3 ``AWC_METAR_BASE`` + UA header) is used as a graceful fallback (D-11);
    its °C ``temp`` field is converted via :func:`celsius_to_fahrenheit`. Endpoints are fixed
    constants (SSRF guard T-02-11). Returns the raw rows; :func:`daily_high` does the bucket.
    """
    station = get_city(city).cli_station
    # IEM strips the leading 'K' from the 4-letter ICAO for its 3-letter network ids.
    iem_station = station[1:] if station.startswith("K") and len(station) == 4 else station

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=15.0, headers={"User-Agent": _USER_AGENT})
    try:
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
            rows = _parse_iem_csv(resp.text)
            if rows:
                return rows
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("IEM ASOS fetch failed for %s (%s); trying AWC fallback", city, exc)

        # Graceful live fallback (D-11): v3 AWC_METAR_BASE, recent window, °C temps.
        return await _fetch_awc_fallback(station, client)
    finally:
        if owns_client:
            await client.aclose()


async def _fetch_awc_fallback(
    station: str, client: httpx.AsyncClient
) -> list[tuple[datetime, float]]:
    """v3 aviationweather.gov METAR fallback (recent ~14h window, °C temps → °F)."""
    try:
        resp = await client.get(
            _AWC_METAR_BASE,
            params={"ids": station, "format": "json", "hours": 14},
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
        if temp_c is None or ts is None:
            continue
        try:
            rows.append((ts, celsius_to_fahrenheit(float(temp_c))))
        except (TypeError, ValueError):
            continue
    return rows


def store_daily_high(
    bind: Bind,
    city: str,
    target_date: date,
    result: DailyHigh,
) -> int:
    """Persist a daily-high label via the SINGLE audited writer path (D-10/D-11).

    Routes through :func:`weatherquant.ingest.writer.insert_observation` with ``source='asos'``
    — never a hand-rolled Core insert. ``available_at`` is the obs REPORT time (the reading
    that produced the max), NOT ``now()`` and NOT the window edge (D-09); the CLI disagreement
    flag is preserved in ``detail`` so the data-quality event is queryable (D-16). Falls back
    to ``window_end`` for ``available_at`` only when no in-window reading exists (empty label).

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
    "DailyHigh",
    "daily_high",
    "daily_high_from_obs",
    "fetch_asos_obs",
    "store_daily_high",
    "celsius_to_fahrenheit",
    "SOURCE",
]
