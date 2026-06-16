"""The ONE ingestion code path (ING-08, D-11/D-15) — compose every Wave 1/2 source.

This module is the integration seam that proves the whole spine runs end to end. It
composes the GRIB core (02-02 :mod:`weatherquant.ingest.grib`), the ground-truth obs +
AFD path (02-03 :mod:`weatherquant.ingest.obs` / :mod:`weatherquant.ingest.afd`), and the
supplementary forecast sources (02-04 :mod:`weatherquant.ingest.sources`) onto a SINGLE
code path that both the live scheduler (:mod:`weatherquant.scheduler`, ``mode="live"``)
and the historical backfill CLI (:mod:`weatherquant.cli`, ``mode="backfill"``) call. Live
vs backfill differ ONLY in the ``available_at`` mode passed through to
:func:`weatherquant.ingest.available_at.available_at` (D-15/D-09) — there is no second,
drifting ingestion path the Phase-6 backtest could diverge from.

THE GRACEFUL-DEGRADATION CONTRACT (D-11). Every source is wrapped in its OWN try/except.
A late/missing model cycle, a failed HTTP fetch, or a decode error emits a STRUCTURED
logged fallback event ``(model, city, cycle, reason)`` and ingestion PROCEEDS with the
other sources — the city/day is NEVER dropped. Crucially, a missing datum is represented
by its ABSENCE: this module never synthesises, back-fills from a neighbour, or otherwise
fabricates a forecast to paper over a gap (D-11). Absence = absence (proven by the
graceful-degradation test, which asserts no row lands for a failed model).

THE OFF-LOOP GRIB DECODE (D-14). The Herbie byte-range fetch + cfgrib decode in
:func:`weatherquant.ingest.grib.fetch_t2m` is sync and CPU-bound; it is run in a thread
executor (``loop.run_in_executor``) so it never blocks the async loop that the HTTP
sources and the AFD fetch share. The HTTP sources are awaited concurrently
(``asyncio.gather``).

CITY VALIDATION (ASVS V5). Every entry point validates the city code via
:func:`weatherquant.registry.get_city`, which raises ``KeyError`` on an unknown code —
never a silent default (T-02-17). Dates are validated at the CLI boundary (02-05 Task 2).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Literal

from weatherquant.ingest import afd as afd_mod
from weatherquant.ingest import grib, obs
from weatherquant.ingest.available_at import available_at
from weatherquant.ingest.sources import nws, openmeteo, wethr
from weatherquant.ingest.writer import insert_forecast
from weatherquant.registry import CITIES, get_city
from weatherquant.time import settlement_window

logger = logging.getLogger(__name__)

Mode = Literal["backfill", "live"]

# The four NOAA models decoded from GRIB byte-range subsets (02-02). HRRR/GFS/NBM are
# deterministic (member=0); GEFS is the 31-member ensemble (c00 + p01..p30).
GRIB_MODELS: tuple[str, ...] = ("nbm", "hrrr", "gfs", "gefs")
_GEFS_MEMBERS: tuple[str, ...] = ("c00",) + tuple(f"p{i:02d}" for i in range(1, 31))

# The supplementary forecast sources (02-04) and the obs/AFD ground-truth path (02-03).
# These are the labels logged in the structured fallback events; they are NOT NOAA models.
SUPPLEMENTARY_SOURCES: tuple[str, ...] = ("nws", "openmeteo", "wethr")


def _gefs_members(model: str) -> tuple[str, ...]:
    """Member labels to fetch for a GRIB model — GEFS is the ensemble, others deterministic."""
    return _GEFS_MEMBERS if model == "gefs" else ("c00",)


def _target_date_for(city_code: str, cycle_init: datetime, lead: int) -> date:
    """Resolve the LST settlement date a (cycle_init + lead) valid instant falls into.

    The valid time is ``cycle_init + lead`` hours (UTC). We assign it to the city's
    settlement (civil) date by walking the candidate days' half-open
    :func:`settlement_window` and returning the one that contains the valid instant. This
    keeps a forecast row's ``target_date`` consistent with the LST window the obs path
    labels against (D-16) — never a hand-rolled UTC day.
    """
    city = get_city(city_code)
    valid = cycle_init + timedelta(hours=lead)
    if valid.tzinfo is None:
        valid = valid.replace(tzinfo=timezone.utc)
    # The valid instant is within ~1 civil day of its UTC date for any US offset; check the
    # UTC date and its neighbours so the half-open window assignment is exact.
    base = valid.astimezone(timezone.utc).date()
    for candidate in (base - timedelta(days=1), base, base + timedelta(days=1)):
        win = settlement_window(city, candidate)
        if win.start_utc <= valid < win.end_utc:
            return candidate
    return base


async def _ingest_grib_model(
    bind: object,
    model: str,
    city_code: str,
    cycle_init: datetime,
    *,
    mode: Mode,
    lead: int,
) -> int:
    """Fetch + decode + snap + write one GRIB model for one city/cycle (off-loop decode, D-14).

    Runs the sync :func:`grib.fetch_t2m` in a thread executor so the cfgrib decode never
    blocks the async loop. For GEFS, iterates every member (c00 + p01..p30), writing each as
    its own ``forecasts`` row keyed by the integer member axis (D-05). Each forecast's
    ``available_at`` is stamped via :func:`available_at` with ``mode`` — the ONLY difference
    between live and backfill (D-15/D-09). Returns the count of rows actually inserted.

    Raises on a fetch/decode failure — the caller wraps this in the per-source try/except so
    a missing cycle degrades gracefully (D-11); this function itself never fabricates a row.
    """
    city = get_city(city_code)
    target_date = _target_date_for(city_code, cycle_init, lead)
    loop = asyncio.get_running_loop()
    stamp = available_at(cycle_init, model, mode)

    inserted = 0
    for member_label in _gefs_members(model):
        # D-14: the sync Herbie+cfgrib decode is CPU-bound — run it OFF the async loop in a
        # thread executor so concurrent HTTP sources are not blocked.
        field = await loop.run_in_executor(
            None, grib.fetch_t2m, model, cycle_init, lead, member_label
        )
        temp_kelvin, station_lat, station_lon, grid_distance_m = grib.snap_city(
            field, city_code
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
    """Emit the ONE structured graceful-degradation fallback event (D-11).

    A late/missing cycle or a fetch error is EXPECTED operation, not a crash: the event is
    logged with ``(model, city, cycle, reason)`` and ingestion proceeds with the other
    sources. No row is written for the failed source — absence = absence (D-11).
    """
    logger.warning(
        "ingest fallback: source=%s city=%s cycle=%s skipped (reason=%s) — "
        "proceeding with other sources, NO fabricated row (D-11)",
        model,
        city,
        cycle.isoformat(),
        reason,
    )


async def ingest_cycle(
    bind: object,
    model: str,
    city: str,
    cycle_init: datetime,
    *,
    mode: Mode,
    lead: int,
) -> int:
    """Ingest a single (model, city, cycle) on the ONE code path (ING-08, D-11/D-15).

    Dispatches to the right fetcher by ``model``:

    * ``nbm`` / ``hrrr`` / ``gfs`` / ``gefs`` — the GRIB byte-range path (02-02), decoded in
      a thread executor (D-14), snapped to the city's Kalshi station, ``available_at`` stamped
      with ``mode`` (D-09).
    * ``nws`` / ``openmeteo`` / ``wethr`` — the supplementary async sources (02-04), live-
      forward only.

    The city is validated via :func:`get_city` (ASVS V5 — raises on an unknown code). The
    whole dispatch is wrapped in a single try/except: any failure logs a STRUCTURED fallback
    and returns 0 (the day is NOT dropped, no fabricated row — D-11). Idempotency (02-02)
    makes a re-run of a completed cycle a no-op (returns 0 inserted).

    Returns:
        The number of ``forecasts`` rows actually inserted (0 on a graceful skip / no-op).
    """
    get_city(city)  # ASVS V5: validate the city code up front; raises on unknown.
    try:
        if model in GRIB_MODELS:
            return await _ingest_grib_model(
                bind, model, city, cycle_init, mode=mode, lead=lead
            )
        if model == "nws":
            return await _ingest_nws(bind, city, cycle_init, lead)
        if model == "openmeteo":
            return await _ingest_openmeteo(bind, city, cycle_init, lead)
        if model == "wethr":
            return await _ingest_wethr(bind, city, cycle_init)
        raise ValueError(f"unknown ingest model/source: {model!r}")
    except KeyError:
        # An unknown city code (ASVS V5) is a caller error, not a graceful-degradation case —
        # never swallow it into a silent fallback.
        raise
    except Exception as exc:  # noqa: BLE001 - per-source graceful degradation (D-11).
        _log_fallback(model, city, cycle_init, exc)
        return 0


async def _ingest_nws(bind: object, city: str, cycle_init: datetime, lead: int) -> int:
    """NWS gridpoint -> in-window max -> one ``nws`` forecast row (02-04, live-forward)."""
    target_date = _target_date_for(city, cycle_init, lead)
    temp_kelvin = await nws.fetch_nws_forecast(city, target_date)
    if temp_kelvin is None:
        _log_fallback("nws", city, cycle_init, "no in-window NWS forecast")
        return 0
    return nws.store_nws_forecast(bind, city, target_date, temp_kelvin, cycle=cycle_init)


async def _ingest_openmeteo(
    bind: object, city: str, cycle_init: datetime, lead: int
) -> int:
    """Open-Meteo ensemble -> per-member rows (02-04, live-forward)."""
    target_date = _target_date_for(city, cycle_init, lead)
    members = await openmeteo.fetch_openmeteo_ensemble(city, target_date)
    if not members:
        _log_fallback("openmeteo", city, cycle_init, "no in-window ensemble members")
        return 0
    return openmeteo.store_members(bind, city, target_date, members, cycle=cycle_init)


async def _ingest_wethr(bind: object, city: str, cycle_init: datetime) -> int:
    """Wethr deterministic models -> per-model rows; graceful skip when key unset (02-04)."""
    target_date = _target_date_for(city, cycle_init, 0)
    inserted = 0
    for model in wethr.WETHR_MODELS:
        temp_kelvin = await wethr.fetch_wethr_forecast(city, model, target_date)
        if temp_kelvin is None:
            # Absent key / no row for this model is a graceful skip (logged inside fetch);
            # absence = absence (D-11), no fabricated row.
            continue
        inserted += wethr.store_wethr_forecast(
            bind, city, model, target_date, temp_kelvin, cycle=cycle_init
        )
    return inserted


async def ingest_obs(
    bind: object, city: str, target_date: date, *, cli_max_f: float | None = None
) -> int:
    """Ground-truth ASOS daily-high for ``city``/``target_date`` (02-03, D-16).

    Fetches sub-daily ASOS/METAR over the LST settlement window, buckets to the daily-high
    label, and stores ``source='asos'`` via the audited writer. A fetch failure logs a
    structured fallback and returns 0 — never a fabricated label (D-11).
    """
    try:
        win = settlement_window(get_city(city), target_date)
        rows = await obs.fetch_asos_obs(city, win)
        result = obs.daily_high(rows, city, target_date, cli_max_f=cli_max_f)
        return obs.store_daily_high(bind, city, target_date, result)
    except KeyError:
        raise
    except Exception as exc:  # noqa: BLE001 - graceful degradation (D-11).
        _log_fallback("asos", city, target_date_to_dt(target_date), exc)
        return 0


async def ingest_afd(
    bind: object, city: str, target_date: date, *, available: datetime | None = None
) -> int:
    """AFD forecaster-disagreement signal for ``city``/``target_date`` (02-03, D-13).

    Fetches the latest AFD product for the city's WFO, runs the v3 keyword pre-filter before
    any paid call, forces the Anthropic ``record_afd_signal`` tool (or degrades to no-signal
    when the key is unset), and stores ``source='afd'``. A fetch/classify failure logs a
    structured fallback and returns 0 (D-11).
    """
    try:
        wfo = afd_mod.CITY_WFO[city]
        text = await afd_mod.fetch_afd_text(wfo)
        if not text:
            _log_fallback("afd", city, target_date_to_dt(target_date), "no AFD product text")
            return 0
        signal = afd_mod.classify_afd(text, wfo)
        return afd_mod.store_afd_signal(
            bind, city, target_date, signal, available_at=available
        )
    except KeyError:
        raise
    except Exception as exc:  # noqa: BLE001 - graceful degradation (D-11).
        _log_fallback("afd", city, target_date_to_dt(target_date), exc)
        return 0


def target_date_to_dt(target_date: date) -> datetime:
    """Coerce a ``date`` to a UTC-midnight ``datetime`` for structured fallback logging."""
    return datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)


async def ingest_all_models(
    bind: object,
    city: str,
    cycle_init: datetime,
    *,
    mode: Mode = "live",
    lead: int = 0,
    models: Sequence[str] | None = None,
) -> dict[str, int]:
    """Ingest EVERY forecast source for one city/cycle on the one path (ING-08, D-11).

    This is the graceful-degradation entry point (turns ``test_graceful_degradation.py``
    GREEN). It calls :func:`ingest_cycle` for each model/source CONCURRENTLY; each is wrapped
    so a single failed/missing model logs a structured fallback and the OTHERS still ingest —
    the city/cycle is never dropped, and no fabricated row is written for the failed model
    (absence = absence, D-11). Returns ``{model: rows_inserted}`` (0 for a skipped source).

    Args:
        bind: a SQLAlchemy Engine/Connection (the single audited writer target).
        city: Kalshi city code (validated via :func:`get_city`).
        cycle_init: the model run init time (UTC).
        mode: ``"live"`` (scheduler) or ``"backfill"`` (CLI) — the ONLY live/backfill seam.
        lead: forecast lead hours for the GRIB models.
        models: optional subset of model/source labels; defaults to all GRIB + supplementary.
    """
    get_city(city)  # ASVS V5 up front.
    targets = list(models) if models is not None else [*GRIB_MODELS, *SUPPLEMENTARY_SOURCES]

    async def _one(model: str) -> tuple[str, int]:
        return model, await ingest_cycle(bind, model, city, cycle_init, mode=mode, lead=lead)

    results = await asyncio.gather(*(_one(m) for m in targets))
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
    bind: object,
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

    Loops every day in ``[start_date, end_date]`` and every requested model cycle, calling
    :func:`ingest_cycle` with ``mode`` (default ``"backfill"`` so ``available_at =
    cycle_init + PUBLISH_LATENCY`` — NEVER ``now()`` for a historical row, D-09). Idempotency
    (02-02) makes re-running a completed range a no-op (each duplicate cycle returns 0
    inserted, D-10). Validates every city via :func:`get_city` (ASVS V5) BEFORE any fetch.

    When ``include_obs`` is set (default), the ground-truth ASOS daily-high and the AFD signal
    are ALSO ingested once per city/day (the obs/AFD path is keyed by settlement date, not by
    cycle) — so a backfill lands the verifying truth alongside the forecasts. Their counts are
    reported under the ``"asos"`` and ``"afd"`` keys.

    Args:
        models: model/source labels to ingest (GRIB and/or supplementary).
        cities: Kalshi city codes (each validated up front; unknown raises KeyError).
        start_date/end_date: inclusive LST settlement date range.
        mode: ``"backfill"`` (default) or ``"live"`` — the single live/backfill seam (D-15).
        lead: forecast lead hours for the GRIB models.
        cycle_hours: UTC cycle init hours to fetch per day (default ``[0]`` — the 00Z run).
        include_obs: also ingest the ASOS daily-high + AFD signal per city/day (default True).

    Returns:
        ``{model: total_rows_inserted}`` across the whole range (plus ``asos``/``afd`` when
        ``include_obs``).
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
                day.year, day.month, day.day, hour, tzinfo=timezone.utc
            )
            for city in cities:
                for model in models:
                    inserted = await ingest_cycle(
                        bind, model, city, cycle_init, mode=mode, lead=lead
                    )
                    totals[model] += inserted
        # The obs/AFD ground truth is keyed by the settlement DATE (not the cycle), so it is
        # ingested once per city/day after the day's forecast cycles (D-16/D-13).
        if include_obs:
            for city in cities:
                totals["asos"] += await ingest_obs(bind, city, day)
                totals["afd"] += await ingest_afd(bind, city, day)
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


ALL_CITIES: tuple[str, ...] = tuple(CITIES)


__all__ = [
    "GRIB_MODELS",
    "SUPPLEMENTARY_SOURCES",
    "ALL_CITIES",
    "ingest_cycle",
    "ingest_range",
    "ingest_all_models",
    "ingest_obs",
    "ingest_afd",
]
