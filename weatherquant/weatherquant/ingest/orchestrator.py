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
from weatherquant.ingest.errors import CorrectnessError, TargetDateError
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
    labels against (D-16) — never a hand-rolled UTC day. If no candidate window contains the
    valid instant (a broken offset/window — never expected for a US offset) this RAISES
    ``ValueError`` rather than silently substituting the raw UTC date (WR-04 / D-16).
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
    # No candidate window contains the valid instant. For any normal US offset the loop above
    # ALWAYS matches, so reaching here means the offset/window math is broken — a real bug, not
    # a backfillable gap. Silently substituting the raw UTC date (the v3 "hand-rolled UTC day"
    # anti-pattern, D-16) would mislabel target_date and the obs path would never join against
    # it. Fail loud instead (WR-04). NEW-1: this raise is a TargetDateError (a CorrectnessError)
    # NOT a bare ValueError — _target_date_for runs INSIDE ingest_cycle's try, so a bare
    # ValueError was caught and downgraded to a silent skip, neutralizing the fail-loud guard.
    raise TargetDateError(
        f"no settlement window contains valid instant {valid.isoformat()} "
        f"for city={city_code} (cycle_init={cycle_init.isoformat()}, lead={lead}) — "
        f"offset/window math is broken; refusing a hand-rolled UTC target_date (D-16)"
    )


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


def _log_live_only_skip(source: str, city: str, cycle: datetime) -> None:
    """Structured skip for a live-only HTTP source requested in backfill (WR-02, D-11).

    nws/openmeteo/wethr return the CURRENT forecast only — they have no point-in-time
    historical archive (the deep corpus is NOAA GRIB). Running them during a historical
    backfill would write today's forecast under a PAST ``target_date`` (fabricated/leaky data,
    D-09). So backfill SKIPS them and logs this event; absence = absence (D-11), no row lands.
    """
    logger.info(
        "ingest skip (backfill): source=%s city=%s cycle=%s is live-only (no point-in-time "
        "historical archive) — NOT run in backfill, NO fabricated row (WR-02/D-11)",
        source,
        city,
        cycle.isoformat(),
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
        if model in SUPPLEMENTARY_SOURCES:
            # The supplementary HTTP sources (nws/openmeteo/wethr) return the CURRENT forecast
            # only — they have no point-in-time historical archive. Running them in backfill
            # would write today's forecast under a PAST target_date (fabricated/leaky data,
            # D-09/D-11). Refuse + structured skip in backfill (WR-02); run normally in live.
            if mode == "backfill":
                _log_live_only_skip(model, city, cycle_init)
                return 0
            if model == "nws":
                return await _ingest_nws(bind, city, cycle_init, lead, mode=mode)
            if model == "openmeteo":
                return await _ingest_openmeteo(bind, city, cycle_init, lead, mode=mode)
            return await _ingest_wethr(bind, city, cycle_init, mode=mode)
        raise ValueError(f"unknown ingest model/source: {model!r}")
    except KeyError:
        # An unknown city code (ASVS V5) is a caller error, not a graceful-degradation case —
        # never swallow it into a silent fallback.
        raise
    except (CorrectnessError, AssertionError):
        # WR-05 / NEW-1: correctness ALARMS must NOT be downgraded to a silent "missing cycle"
        # skip. CorrectnessError covers ALL of them by base class — the writer's rowcount
        # integrity breach (WriteIntegrityError), a unit mismatch (UnitError), a lead-0 / snap
        # breach (SanityError), and the impossible-window target_date raise (TargetDateError) —
        # plus any AssertionError. These are real bugs: let them propagate LOUDLY so they page,
        # rather than vanishing into a graceful fallback that masks corruption (D-11). (The
        # earlier version caught only WriteIntegrityError/AssertionError, so the bare-ValueError
        # alarms — and WR-04's own raise — were still swallowed below: WR-05 partial + NEW-1.)
        raise
    except Exception as exc:  # noqa: BLE001 - per-source graceful degradation (D-11).
        # EXPECTED transient/degradation failures ONLY: a late/missing model cycle, an HTTP
        # fetch error (httpx), a Herbie/cfgrib decode error, a timeout/connection error. These
        # log a STRUCTURED fallback and the day proceeds with the other sources. The correctness
        # alarms above are excluded by type, so a real bug can never hide here.
        _log_fallback(model, city, cycle_init, exc)
        return 0


async def _ingest_nws(
    bind: object, city: str, cycle_init: datetime, lead: int, *, mode: Mode
) -> int:
    """NWS gridpoint -> in-window max -> one ``nws`` forecast row (02-04, live-forward)."""
    target_date = _target_date_for(city, cycle_init, lead)
    temp_kelvin = await nws.fetch_nws_forecast(city, target_date)
    if temp_kelvin is None:
        _log_fallback("nws", city, cycle_init, "no in-window NWS forecast")
        return 0
    return nws.store_nws_forecast(
        bind, city, target_date, temp_kelvin, cycle=cycle_init, mode=mode
    )


async def _ingest_openmeteo(
    bind: object, city: str, cycle_init: datetime, lead: int, *, mode: Mode
) -> int:
    """Open-Meteo ensemble -> per-member rows (02-04, live-forward)."""
    target_date = _target_date_for(city, cycle_init, lead)
    members = await openmeteo.fetch_openmeteo_ensemble(city, target_date)
    if not members:
        _log_fallback("openmeteo", city, cycle_init, "no in-window ensemble members")
        return 0
    return openmeteo.store_members(
        bind, city, target_date, members, cycle=cycle_init, mode=mode
    )


async def _ingest_wethr(
    bind: object, city: str, cycle_init: datetime, *, mode: Mode
) -> int:
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
            bind, city, model, target_date, temp_kelvin, cycle=cycle_init, mode=mode
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
    except (CorrectnessError, AssertionError):
        # WR-05/NEW-1: any correctness alarm on the obs write path (rowcount/unit/sanity/
        # target_date) is a real bug — propagate loudly, do not degrade to a silent skip.
        raise
    except Exception as exc:  # noqa: BLE001 - graceful degradation (D-11).
        _log_fallback("asos", city, target_date_to_dt(target_date), exc)
        return 0


async def ingest_afd(
    bind: object,
    city: str,
    target_date: date,
    *,
    available: datetime | None = None,
    mode: Mode = "live",
) -> int:
    """AFD forecaster-disagreement signal for ``city``/``target_date`` (02-03, D-13).

    Fetches the latest AFD product for the city's WFO, runs the v3 keyword pre-filter before
    any paid call, forces the Anthropic ``record_afd_signal`` tool (or degrades to no-signal
    when the key is unset), and stores ``source='afd'``. A fetch/classify failure logs a
    structured fallback and returns 0 (D-11).

    POINT-IN-TIME INTEGRITY (CR-01, D-09). An AFD row's ``available_at`` MUST be the product
    ISSUANCE time, never ``now()``. ``fetch_afd_text`` returns only the LATEST (current)
    product, so a historical issuance time cannot be recovered in backfill. Therefore in
    ``mode="backfill"`` AFD is SKIPPED unless the caller supplies an explicit ``available``
    issuance time — a structured skip is logged and NO row lands (absence = absence, D-11),
    rather than stamping ``now()`` on a row reconstructed for a past date (the CR-01 leak).

    OFF-LOOP CLASSIFY (WR-03, D-14). ``classify_afd`` can make a blocking Anthropic SDK call;
    it is run in a thread executor (like the GRIB decode) so it never blocks the async loop.
    Paid classification is additionally gated OFF in backfill (no live AFD issuance to stamp,
    and a range backfill must not fan out paid per-city/day calls).
    """
    try:
        wfo = afd_mod.CITY_WFO[city]
        if mode == "backfill" and available is None:
            # CR-01: cannot recover a historical AFD issuance time from the live "latest"
            # product fetch. Refuse to stamp now() on a backfilled row — skip + structured log
            # (absence = absence, D-11). Live mode (or an explicit issuance time) still runs.
            _log_live_only_skip("afd", city, target_date_to_dt(target_date))
            return 0
        text = await afd_mod.fetch_afd_text(wfo)
        if not text:
            _log_fallback("afd", city, target_date_to_dt(target_date), "no AFD product text")
            return 0
        # WR-03/D-14: the classify path may issue a BLOCKING Anthropic SDK call — run it off
        # the event loop in a thread executor so concurrent fetches are not blocked.
        loop = asyncio.get_running_loop()
        signal = await loop.run_in_executor(None, afd_mod.classify_afd, text, wfo)
        return afd_mod.store_afd_signal(
            bind, city, target_date, signal, available_at=available, mode=mode
        )
    except KeyError:
        raise
    except (CorrectnessError, AssertionError):
        # WR-05/NEW-1: any correctness alarm on the AFD path (rowcount integrity, the
        # AvailabilityError backfill-now() guard, target_date, etc.) is a real bug — propagate
        # loudly, do not degrade to a silent skip.
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

    LIVE-ONLY SOURCES ARE NOT BACKFILLED (WR-02). nws/openmeteo/wethr return only the CURRENT
    forecast — they have no point-in-time historical archive — so in ``mode="backfill"``
    :func:`ingest_cycle` SKIPS them with a structured log (absence = absence, D-11) rather than
    writing today's forecast under N past ``target_date``s. Only the GRIB models (NOAA archive)
    and the ASOS obs (IEM archive) carry real historical data; AFD runs only when a historical
    issuance time is recoverable (skipped in plain backfill — CR-01). In ``mode="live"`` every
    source runs normally. This is a behavior change: a range backfill no longer fabricates the
    supplementary/AFD rows it previously mis-stamped.

    When ``include_obs`` is set (default), the ground-truth ASOS daily-high (and AFD in live
    mode) are ALSO ingested once per city/day (the obs/AFD path is keyed by settlement date,
    not by cycle) — so a backfill lands the verifying truth alongside the forecasts. Their
    counts are reported under the ``"asos"`` and ``"afd"`` keys.

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
                # ASOS has a true point-in-time historical archive (IEM) — it is the verifying
                # truth and runs in backfill. AFD is threaded with mode: in backfill it skips
                # (no recoverable historical issuance time — CR-01) unless an issuance time is
                # supplied; in live it stamps the report time.
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
