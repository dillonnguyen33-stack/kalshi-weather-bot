"""The ONE ingestion code path (ING-08, D-11/D-15) — compose every Wave 1/2 source.

Composes the GRIB core, obs/AFD ground truth, and supplementary sources onto a single
path both the live scheduler and the backfill CLI call; live vs backfill differ only in
the ``available_at`` mode (D-15/D-09; see docs/DECISIONS.md). Each source is wrapped in its
own try/except for graceful degradation, the off-loop GRIB decode runs in a thread executor,
and every entry point validates the city via :func:`get_city` (D-11/D-14; ASVS V5).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import date, datetime, timedelta, UTC
from typing import Literal

import httpx

from weatherquant.ingest import afd as afd_mod
from weatherquant.ingest import grib, obs
from weatherquant.ingest.available_at import available_at
from weatherquant.ingest.errors import CorrectnessError, TargetDateError
from weatherquant.ingest.sources import nws, openmeteo, wethr
from weatherquant.ingest.sources._client import get_client
from weatherquant.ingest.writer import Bind, insert_forecast
from weatherquant.registry import get_city
from weatherquant.time import settlement_window

logger = logging.getLogger(__name__)

Mode = Literal["backfill", "live"]

# The four NOAA models decoded from GRIB byte-range subsets (02-02). HRRR/GFS/NBM are
# deterministic (member=0); GEFS is the 31-member ensemble (c00 + p01..p30).
GRIB_MODELS: tuple[str, ...] = ("nbm", "hrrr", "gfs", "gefs")
_GEFS_MEMBERS: tuple[str, ...] = ("c00",) + tuple(f"p{i:02d}" for i in range(1, 31))

# Snap-distance sane bound (Pitfall 2) wired into the live path so grib.snap_city's SanityError
# can actually fire. Covers the coarsest grid we decode (GEFS 0.5° ≈ 56 km nearest cell) with
# headroom; a bad station coord / grid mislabel snaps thousands of km off, far past this.
# ponytail: one bound for all models; tighten per-model only if a mislabel ever slips under 80km.
_MAX_SNAP_DISTANCE_M = 80_000.0

# The supplementary live-forward sources (02-04) are defined as the {name: handler} dispatch
# table _SUPPLEMENTARY_HANDLERS below (the single source of truth); SUPPLEMENTARY_SOURCES is
# derived from its keys so the set, the ingest_cycle dispatch, and the CLI selector never drift.


def _gefs_members(model: str) -> tuple[str, ...]:
    """Member labels to fetch for a GRIB model — GEFS is the ensemble, others deterministic."""
    return _GEFS_MEMBERS if model == "gefs" else ("c00",)


def _target_date_for(city_code: str, cycle_init: datetime, lead: int) -> date:
    """Resolve the LST settlement date the (cycle_init + lead) valid instant falls into (D-16).

    Walks candidate days' half-open :func:`settlement_window` so ``target_date`` matches the
    LST window the obs path labels against; an impossible window raises (WR-04; see
    docs/DECISIONS.md).
    """
    city = get_city(city_code)
    valid = cycle_init + timedelta(hours=lead)
    if valid.tzinfo is None:
        valid = valid.replace(tzinfo=UTC)
    # Valid instant is within ~1 civil day of its UTC date for any US offset; check the UTC
    # date and its neighbours so the half-open window assignment is exact.
    base = valid.astimezone(UTC).date()
    for candidate in (base - timedelta(days=1), base, base + timedelta(days=1)):
        win = settlement_window(city, candidate)
        if win.start_utc <= valid < win.end_utc:
            return candidate
    # No window matched: broken offset/window math, fail loud rather than mislabel target_date
    # (WR-04; see docs/DECISIONS.md). NEW-1: TargetDateError (a CorrectnessError), NOT a bare
    # ValueError, so ingest_cycle's try cannot downgrade this fail-loud guard to a silent skip.
    raise TargetDateError(
        f"no settlement window contains valid instant {valid.isoformat()} "
        f"for city={city_code} (cycle_init={cycle_init.isoformat()}, lead={lead}) — "
        f"offset/window math is broken; refusing a hand-rolled UTC target_date (D-16)"
    )


async def _ingest_grib_model(
    bind: Bind,
    model: str,
    city_code: str,
    cycle_init: datetime,
    *,
    mode: Mode,
    lead: int,
) -> int:
    """Fetch + decode + snap + write one GRIB model for one city/cycle (off-loop decode, D-14).

    Iterates GEFS members (c00 + p01..p30) as separate rows keyed by the member axis (D-05);
    ``available_at`` is stamped with ``mode`` (D-15/D-09). Returns rows inserted. Raises on a
    fetch/decode failure for the caller's per-source try/except (D-11).
    """
    city = get_city(city_code)
    target_date = _target_date_for(city_code, cycle_init, lead)
    loop = asyncio.get_running_loop()
    stamp = available_at(cycle_init, model, mode)

    inserted = 0
    for i, member_label in enumerate(_gefs_members(model)):
        # D-14: run the sync, CPU-bound Herbie+cfgrib decode off the async loop.
        field = await loop.run_in_executor(
            None, grib.fetch_t2m, model, cycle_init, lead, member_label
        )
        temp_kelvin, station_lat, station_lon, grid_distance_m = grib.snap_city(
            field, city_code, max_distance_m=_MAX_SNAP_DISTANCE_M
        )
        # D-04 / Pitfall 4: the lead-0 control snap must track contemporaneous ASOS, else a
        # wrong snap / unit / grid error fails LOUD (SanityError → ingest_cycle re-raises, never
        # a silent skip that corrupts every downstream fit). Checked once per model on the c00
        # control member (i==0); the grid is shared across members so one probe covers them all.
        # ASOS unavailable → skip the probe (can't verify), only a real breach raises.
        if lead == 0 and i == 0:
            asos_k = await obs.asos_lead0_kelvin(city_code, target_date, cycle_init)
            if asos_k is not None:
                grib.lead0_sanity_check(
                    forecast_k=temp_kelvin, asos_k=asos_k, city_code=city_code
                )
        inserted += insert_forecast(
            bind,
            city=city_code,
            target_date=target_date,
            model=model,
            lead=lead,
            member=grib.member_to_axis(member_label),
            temp_kelvin=temp_kelvin,
            cycle=cycle_init,
            station_lat=station_lat,
            station_lon=station_lon,
            grid_distance_m=grid_distance_m,
            available_at=stamp,
        )
    logger.debug(
        "grib ingest model=%s city=%s cycle=%s lead=%s -> %d row(s) (station=%s)",
        model,
        city_code,
        cycle_init.isoformat(),
        lead,
        inserted,
        city.cli_station,
    )
    return inserted


def _log_fallback(model: str, city: str, cycle: datetime, reason: object) -> None:
    """Emit the structured graceful-degradation fallback event for an expected miss (D-11)."""
    logger.warning(
        "ingest fallback: source=%s city=%s cycle=%s skipped (reason=%s) — "
        "proceeding with other sources, NO fabricated row (D-11)",
        model,
        city,
        cycle.isoformat(),
        reason,
    )


def _log_live_only_skip(source: str, city: str, cycle: datetime) -> None:
    """Structured skip for a live-only HTTP source requested in backfill (WR-02, D-11).

    nws/openmeteo/wethr have no point-in-time archive, so backfilling them would stamp today's
    forecast under a past ``target_date`` (D-09; see docs/DECISIONS.md). Backfill skips + logs.
    """
    logger.info(
        "ingest skip (backfill): source=%s city=%s cycle=%s is live-only (no point-in-time "
        "historical archive) — NOT run in backfill, NO fabricated row (WR-02/D-11)",
        source,
        city,
        cycle.isoformat(),
    )


async def ingest_cycle(
    bind: Bind,
    model: str,
    city: str,
    cycle_init: datetime,
    *,
    mode: Mode,
    lead: int,
    client: httpx.AsyncClient | None = None,
) -> int:
    """Ingest a single (model, city, cycle) on the ONE code path, dispatching by model (ING-08).

    GRIB models take the byte-range path; nws/openmeteo/wethr take the live-forward async
    sources. Wrapped in one try/except: any transient failure logs a fallback and returns 0
    (D-11/D-15; see docs/DECISIONS.md).

    Returns:
        ``forecasts`` rows inserted (0 on a graceful skip / no-op).
    """
    get_city(city)  # ASVS V5: validate the city code up front; raises on unknown.
    try:
        if model in GRIB_MODELS:
            return await _ingest_grib_model(
                bind, model, city, cycle_init, mode=mode, lead=lead
            )
        handler = _SUPPLEMENTARY_HANDLERS.get(model)
        if handler is None:
            raise ValueError(f"unknown ingest model/source: {model!r}")
        # Live-forward sources have no historical archive: skip in backfill, run in live
        # (WR-02 / D-09 / D-11; see docs/DECISIONS.md).
        if mode == "backfill":
            _log_live_only_skip(model, city, cycle_init)
            return 0
        return await handler(bind, city, cycle_init, lead, mode=mode, client=client)
    except KeyError:
        # Unknown city code (ASVS V5) is a caller error, not graceful degradation — never swallow.
        raise
    except (CorrectnessError, AssertionError):
        # Correctness ALARMS re-raise loudly, never downgrade to a silent skip (WR-05/NEW-1):
        # the base class covers all of them so a real bug can't hide in the fallback below.
        raise
    except Exception as exc:  # noqa: BLE001 - per-source graceful degradation (D-11).
        # EXPECTED transient failures only (late cycle, HTTP/decode/timeout); alarms above are
        # excluded by type, so degrading here can never mask corruption.
        _log_fallback(model, city, cycle_init, exc)
        return 0


async def _ingest_nws(
    bind: Bind, city: str, cycle_init: datetime, lead: int, *, mode: Mode,
    client: httpx.AsyncClient | None = None,
) -> int:
    """NWS gridpoint -> in-window max -> one ``nws`` forecast row (02-04, live-forward)."""
    target_date = _target_date_for(city, cycle_init, lead)
    temp_kelvin = await nws.fetch_nws_forecast(city, target_date, client=client)
    if temp_kelvin is None:
        _log_fallback("nws", city, cycle_init, "no in-window NWS forecast")
        return 0
    return nws.store_nws_forecast(
        bind, city, target_date, temp_kelvin, cycle=cycle_init, mode=mode
    )


async def _ingest_openmeteo(
    bind: Bind, city: str, cycle_init: datetime, lead: int, *, mode: Mode,
    client: httpx.AsyncClient | None = None,
) -> int:
    """Open-Meteo ensemble -> per-member rows (02-04, live-forward)."""
    target_date = _target_date_for(city, cycle_init, lead)
    members = await openmeteo.fetch_openmeteo_ensemble(city, target_date, client=client)
    if not members:
        _log_fallback("openmeteo", city, cycle_init, "no in-window ensemble members")
        return 0
    return openmeteo.store_members(
        bind, city, target_date, members, cycle=cycle_init, mode=mode
    )


async def _ingest_wethr(
    bind: Bind, city: str, cycle_init: datetime, lead: int, *, mode: Mode,
    client: httpx.AsyncClient | None = None,
) -> int:
    """Wethr deterministic models -> per-model rows; graceful skip when key unset (02-04).

    ``lead`` is accepted for the uniform dispatch signature but Wethr serves the latest run only,
    so rows are keyed at lead 0 (the cosmetic asymmetry is intentional — all callers pass 0).
    """
    target_date = _target_date_for(city, cycle_init, 0)
    inserted = 0
    for model in wethr.WETHR_MODELS:
        temp_kelvin = await wethr.fetch_wethr_forecast(city, model, target_date, client=client)
        if temp_kelvin is None:
            continue  # absent key / no row is a graceful skip, logged inside fetch (D-11).
        inserted += wethr.store_wethr_forecast(
            bind, city, model, target_date, temp_kelvin, cycle=cycle_init, mode=mode
        )
    return inserted


# The supplementary live-forward sources: name -> ingest handler, the SINGLE source of truth for
# the supplementary set (02-04). ingest_cycle dispatches through this table and
# SUPPLEMENTARY_SOURCES is derived from its keys, so adding a source is a one-line change here.
# All handlers share ``(bind, city, cycle_init, lead, *, mode, client)``.
_SUPPLEMENTARY_HANDLERS = {
    "nws": _ingest_nws,
    "openmeteo": _ingest_openmeteo,
    "wethr": _ingest_wethr,
}
SUPPLEMENTARY_SOURCES: tuple[str, ...] = tuple(_SUPPLEMENTARY_HANDLERS)


async def ingest_obs(
    bind: Bind, city: str, target_date: date, *, cli_max_f: float | None = None
) -> int:
    """Ground-truth ASOS daily-high for ``city``/``target_date`` via the audited writer (D-16).

    A fetch failure logs a fallback and returns 0, never a fabricated label (D-11).
    """
    try:
        win = settlement_window(get_city(city), target_date)
        rows = await obs.fetch_asos_obs(city, win)
        result = obs.daily_high(rows, city, target_date, cli_max_f=cli_max_f)
        return obs.store_daily_high(bind, city, target_date, result)
    except KeyError:
        raise
    except (CorrectnessError, AssertionError):
        # Correctness alarm on the obs write path re-raises loudly, never degrades (WR-05/NEW-1).
        raise
    except Exception as exc:  # noqa: BLE001 - graceful degradation (D-11).
        _log_fallback("asos", city, target_date_to_dt(target_date), exc)
        return 0


async def ingest_afd(
    bind: Bind,
    city: str,
    target_date: date,
    *,
    available: datetime | None = None,
    mode: Mode = "live",
) -> int:
    """AFD forecaster-disagreement signal for ``city``/``target_date`` via the audited writer (D-13).

    Fetches the latest AFD product, pre-filters before any paid call, forces the Anthropic
    ``record_afd_signal`` tool, and stores ``source='afd'``; failures degrade to 0 (D-11).
    Backfill skips unless an explicit ``available`` issuance time is supplied, since ``now()``
    on a historical row leaks (CR-01/D-09); classify runs off-loop (WR-03/D-14). See
    docs/DECISIONS.md.
    """
    try:
        wfo = afd_mod.CITY_WFO[city]
        if mode == "backfill" and available is None:
            # CR-01: a historical issuance time isn't recoverable from the live "latest" fetch,
            # so refuse to stamp now() on a backfilled row — skip + log (see docs/DECISIONS.md).
            _log_live_only_skip("afd", city, target_date_to_dt(target_date))
            return 0
        text = await afd_mod.fetch_afd_text(wfo)
        if not text:
            _log_fallback("afd", city, target_date_to_dt(target_date), "no AFD product text")
            return 0
        # WR-03/D-14: run the possibly-blocking Anthropic classify call off the event loop.
        loop = asyncio.get_running_loop()
        signal = await loop.run_in_executor(None, afd_mod.classify_afd, text, wfo)
        return afd_mod.store_afd_signal(
            bind, city, target_date, signal, available_at=available, mode=mode
        )
    except KeyError:
        raise
    except (CorrectnessError, AssertionError):
        # Correctness alarm on the AFD path re-raises loudly, never degrades (WR-05/NEW-1).
        raise
    except Exception as exc:  # noqa: BLE001 - graceful degradation (D-11).
        _log_fallback("afd", city, target_date_to_dt(target_date), exc)
        return 0


def target_date_to_dt(target_date: date) -> datetime:
    """Coerce a ``date`` to a UTC-midnight ``datetime`` for structured fallback logging."""
    return datetime(target_date.year, target_date.month, target_date.day, tzinfo=UTC)


async def ingest_all_models(
    bind: Bind,
    city: str,
    cycle_init: datetime,
    *,
    mode: Mode = "live",
    lead: int = 0,
    models: Sequence[str] | None = None,
) -> dict[str, int]:
    """Ingest EVERY forecast source for one city/cycle concurrently, degrading per-source (D-11).

    The graceful-degradation entry point: a failed model logs a fallback while the others still
    ingest. Returns ``{model: rows_inserted}`` (0 for a skipped source).

    Args:
        mode: ``"live"`` (scheduler) or ``"backfill"`` (CLI) — the only live/backfill seam.
        models: optional subset of labels; defaults to all GRIB + supplementary.
    """
    get_city(city)  # ASVS V5 up front.
    targets = list(models) if models is not None else [*GRIB_MODELS, *SUPPLEMENTARY_SOURCES]

    # One shared httpx client for the whole concurrent fan-out (D-14): the supplementary
    # sources reuse it instead of each opening/closing its own; closed once here.
    client = get_client()
    try:

        async def _one(model: str) -> tuple[str, int]:
            return model, await ingest_cycle(
                bind, model, city, cycle_init, mode=mode, lead=lead, client=client
            )

        results = await asyncio.gather(*(_one(m) for m in targets))
    finally:
        await client.aclose()
    summary = dict(results)
    logger.info(
        "ingest_all_models city=%s cycle=%s mode=%s -> %s",
        city,
        cycle_init.isoformat(),
        mode,
        summary,
    )
    return summary


async def ingest_range(
    bind: Bind,
    models: Sequence[str],
    cities: Sequence[str],
    start_date: date,
    end_date: date,
    *,
    mode: Mode = "backfill",
    lead: int = 0,
    cycle_hours: Sequence[int] | None = None,
    include_obs: bool = True,
) -> dict[str, int]:
    """Backfill a date range on the SAME code path the scheduler uses (D-15/D-10).

    Loops each day/cycle through :func:`ingest_cycle` with ``mode`` (default ``"backfill"`` so
    ``available_at = cycle_init + PUBLISH_LATENCY``, never ``now()`` — D-09); idempotency makes
    a re-run a no-op (D-10). Live-only sources (nws/openmeteo/wethr) are skipped in backfill
    (WR-02); AFD runs only when a historical issuance time is recoverable (CR-01). When
    ``include_obs`` (default), the ASOS daily-high (and AFD in live mode) is ingested once per
    city/day — keyed by settlement date, not cycle — reported under ``asos``/``afd``. See
    docs/DECISIONS.md.

    Args:
        cities: Kalshi city codes (validated up front; unknown raises KeyError).
        start_date/end_date: inclusive LST settlement date range.
        mode: ``"backfill"`` (default) or ``"live"`` — the single live/backfill seam (D-15).
        cycle_hours: UTC cycle init hours per day (default ``[0]`` — the 00Z run).
        include_obs: also ingest the ASOS daily-high + AFD signal per city/day (default True).

    Returns:
        ``{model: total_rows_inserted}`` (plus ``asos``/``afd`` when ``include_obs``).
    """
    for city in cities:
        get_city(city)  # ASVS V5: reject an unknown city before any fetch.
    if end_date < start_date:
        raise ValueError(f"end_date {end_date} is before start_date {start_date}")

    hours = list(cycle_hours) if cycle_hours is not None else [0]
    totals: dict[str, int] = {m: 0 for m in models}
    if include_obs:
        totals.setdefault("asos", 0)
        totals.setdefault("afd", 0)

    day = start_date
    while day <= end_date:
        for hour in hours:
            cycle_init = datetime(
                day.year, day.month, day.day, hour, tzinfo=UTC
            )
            for city in cities:
                for model in models:
                    inserted = await ingest_cycle(
                        bind, model, city, cycle_init, mode=mode, lead=lead
                    )
                    totals[model] += inserted
        # Obs/AFD ground truth is keyed by settlement DATE, so it runs once per city/day after
        # the day's cycles (D-16/D-13).
        if include_obs:
            for city in cities:
                # ASOS has an IEM historical archive and runs in backfill; AFD skips in backfill
                # absent a recoverable issuance time (CR-01).
                totals["asos"] += await ingest_obs(bind, city, day)
                totals["afd"] += await ingest_afd(bind, city, day, mode=mode)
        day += timedelta(days=1)

    logger.info(
        "ingest_range %s..%s cities=%s models=%s mode=%s -> %s",
        start_date,
        end_date,
        list(cities),
        list(models),
        mode,
        totals,
    )
    return totals


__all__ = [
    "GRIB_MODELS",
    "SUPPLEMENTARY_SOURCES",
    "ingest_afd",
    "ingest_all_models",
    "ingest_cycle",
    "ingest_obs",
    "ingest_range",
]
