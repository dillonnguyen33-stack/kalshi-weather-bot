"""AsyncIOScheduler wiring (D-15) — the LIVE half of the one ingestion code path.

``build_scheduler`` returns a configured, NOT-yet-started ``AsyncIOScheduler`` whose jobs call
the SAME :func:`weatherquant.ingest.orchestrator.ingest_cycle` the backfill CLI calls, only
with ``mode="live"`` (D-09/D-15) — no second drifting live path the backtest could diverge
from.

APSCHEDULER PIN (T-02-SC): apscheduler **3.11.x** ``AsyncIOScheduler`` — never the 4.x rewrite
(different package / import paths). Shares the asyncio loop with the httpx sources and the
(future) Kalshi WS feed.

PER-MODEL CADENCE: HRRR/NBM hourly; GFS/GEFS the 00/06/12/18Z synoptic cycles; NWS/obs/AFD
hourly, Open-Meteo per-GFS-cycle. Returned UNSTARTED so it is unit-testable and the caller
owns ``start()``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from weatherquant.db.engine import get_engine
from weatherquant.ingest import orchestrator
from weatherquant.registry import CITIES

logger = logging.getLogger(__name__)

# Cycle latency cushion: a cadence fires for the cycle that has had time to publish; the
# orchestrator handles a not-yet-published cycle gracefully (D-11), so clock skew never crashes.

# The live feed ingests TWO leads each cycle: lead 0 (the contemporaneous nowcast + the D-04 ASOS
# sanity snap) and lead 24 (the calibrated/traded 24h-ahead daily high — what pricing reads). lead 0
# is PRIMARY (always published for the cycle); lead 24 is BEST-EFFORT — not every model/cycle
# publishes fxx=24 (e.g. HRRR hourly runs stop at fxx=18), so a missing lead-24 degrades+logs and
# never sinks the guaranteed lead-0 write.
# ponytail: lead 24 on the HRRR-hourly cadence is often absent → routine warnings; move lead 24 to
# the synoptic cadence only if that log noise ever matters.
PRIMARY_LEAD = 0
TRADE_LEAD = 24
LIVE_LEADS = (PRIMARY_LEAD, TRADE_LEAD)


def _both_leads(
    ingest_one: Callable[[str, int], Awaitable[object]],
) -> Callable[[str], Awaitable[None]]:
    """Wrap a per-(city, lead) ingest into a per-city coro that does lead 0 then best-effort lead 24."""

    async def _one(city: str) -> None:
        await ingest_one(city, PRIMARY_LEAD)  # primary: its failure marks the city failed (correct)
        try:
            await ingest_one(city, TRADE_LEAD)
        except Exception as exc:  # noqa: BLE001 - best-effort trade lead; a missing fxx must not sink lead 0
            logger.warning(
                "live ingest city=%s lead=%s unavailable (best-effort, D-11): %r",
                city, TRADE_LEAD, exc,
            )

    return _one


async def _run_per_city(
    make_coro: Callable[[str], Awaitable[object]], label: str
) -> tuple[int, int]:
    """Run one coroutine per city concurrently, degrading PER CITY (D-11), returning (n_ok, n_failed).

    ``orchestrator.ingest_cycle`` deliberately re-raises a ``CorrectnessError`` (an alarm). The
    orchestrator's own fan-out stays bare so alarms propagate, but the SCHEDULER is the wrong
    layer to inherit that: one city's bad window must not abort every other city for the model.
    So gather with ``return_exceptions=True`` and log each failure (alarms stay visible in the
    logs) without letting it kill the job.
    """
    cities = list(CITIES)
    results = await asyncio.gather(
        *(make_coro(city) for city in cities), return_exceptions=True
    )
    n_failed = 0
    for city, result in zip(cities, results):
        if isinstance(result, BaseException):
            n_failed += 1
            logger.error("live job [%s] city=%s failed: %r", label, city, result)
    return len(cities) - n_failed, n_failed


def _latest_synoptic_cycle(now: datetime, step_hours: int) -> datetime:
    """Floor ``now`` to the most recent ``step_hours`` synoptic cycle (e.g. 6h -> 00/06/12/18Z)."""
    floored_hour = (now.hour // step_hours) * step_hours
    return now.replace(hour=floored_hour, minute=0, second=0, microsecond=0)


def _latest_hourly_cycle(now: datetime) -> datetime:
    """Floor ``now`` to the top of the current hour (the latest HRRR/NBM cycle)."""
    return now.replace(minute=0, second=0, microsecond=0)


async def _ingest_grib_all_cities(model: str, step_hours: int) -> None:
    """Live job body: ingest ``model`` for the latest cycle across every registry city (D-15).

    Calls the SAME :func:`orchestrator.ingest_cycle` the CLI backfill uses with ``mode="live"``
    (D-09/D-15); cities are independent under graceful degradation (D-11).
    """
    now = datetime.now(UTC)
    cycle = (
        _latest_hourly_cycle(now) if step_hours == 1 else _latest_synoptic_cycle(now, step_hours)
    )
    bind = get_engine()
    # Cities are independent (per-city graceful degradation, D-11) — run concurrently so the job
    # finishes well within its cadence and never overruns into the next firing. Each city ingests
    # both leads (lead 0 primary, lead 24 best-effort — see LIVE_LEADS).
    n_ok, n_failed = await _run_per_city(
        _both_leads(
            lambda city, lead: orchestrator.ingest_cycle(
                bind, model, city, cycle, mode="live", lead=lead
            )
        ),
        f"grib model={model}",
    )
    logger.info(
        "live grib job model=%s cycle=%s leads=%s ran across %d cities (ok=%d failed=%d)",
        model, cycle, LIVE_LEADS, len(CITIES), n_ok, n_failed,
    )


async def _ingest_source_all_cities(source: str, step_hours: int) -> None:
    """Live job body for a supplementary source (nws/openmeteo) across every city (D-15)."""
    now = datetime.now(UTC)
    cycle = (
        _latest_hourly_cycle(now) if step_hours == 1 else _latest_synoptic_cycle(now, step_hours)
    )
    bind = get_engine()
    n_ok, n_failed = await _run_per_city(
        _both_leads(
            lambda city, lead: orchestrator.ingest_cycle(
                bind, source, city, cycle, mode="live", lead=lead
            )
        ),
        f"source={source}",
    )
    logger.info(
        "live source job source=%s cycle=%s leads=%s ran across %d cities (ok=%d failed=%d)",
        source, cycle, LIVE_LEADS, len(CITIES), n_ok, n_failed,
    )


async def _ingest_obs_all_cities() -> None:
    """Live job body: ASOS daily-high + AFD signal for today across every city (02-03)."""
    today = datetime.now(UTC).date()
    bind = get_engine()

    async def _one(city: str) -> None:
        await orchestrator.ingest_obs(bind, city, today)
        await orchestrator.ingest_afd(bind, city, today)

    n_ok, n_failed = await _run_per_city(_one, "obs/afd")
    logger.info(
        "live obs/afd job ran for %s across %d cities (ok=%d failed=%d)",
        today, len(CITIES), n_ok, n_failed,
    )


# Explicit misfire policy on every job (vs apscheduler's silent 1s default): a firing up to
# 30 min late STILL runs, coalesce collapses a backlog of missed firings into one, and
# max_instances=1 + idempotent re-runs (D-10) keep concurrent same-job runs from double-writing.
# The gather'd bodies finish fast, so a job never overruns into the next cycle.
_JOB_POLICY = {"misfire_grace_time": 1800, "coalesce": True, "max_instances": 1}


def build_scheduler() -> AsyncIOScheduler:
    """Build (but do NOT start) the live ingestion scheduler (D-15, apscheduler 3.11.x).

    One job per cadence, each calling the orchestrator with ``mode="live"``: hrrr/nbm hourly;
    gfs/gefs at 00/06/12/18Z; nws hourly, openmeteo per-GFS-cycle, obs/AFD hourly. Returned
    UNSTARTED so it is unit-testable (``get_jobs()``) and the caller owns ``start()``.
    """
    scheduler = AsyncIOScheduler(timezone="UTC")

    # HRRR / NBM — hourly cycles (top of the hour).
    for model in ("hrrr", "nbm"):
        scheduler.add_job(
            _ingest_grib_all_cities,
            CronTrigger(minute=0, timezone="UTC"),
            args=[model, 1],
            id=f"grib-{model}-hourly",
            name=f"live ingest {model} (hourly)",
            replace_existing=True,
            **_JOB_POLICY,
        )

    # GFS / GEFS — the 00/06/12/18Z synoptic cycles (D-15).
    for model in ("gfs", "gefs"):
        scheduler.add_job(
            _ingest_grib_all_cities,
            CronTrigger(hour="0,6,12,18", minute=0, timezone="UTC"),
            args=[model, 6],
            id=f"grib-{model}-synoptic",
            name=f"live ingest {model} (00/06/12/18Z)",
            replace_existing=True,
            **_JOB_POLICY,
        )

    # NWS gridpoint — hourly; Open-Meteo ensemble — per-GFS cycle.
    scheduler.add_job(
        _ingest_source_all_cities,
        CronTrigger(minute=15, timezone="UTC"),
        args=["nws", 1],
        id="source-nws-hourly",
        name="live ingest nws (hourly)",
        replace_existing=True,
        **_JOB_POLICY,
    )
    scheduler.add_job(
        _ingest_source_all_cities,
        CronTrigger(hour="0,6,12,18", minute=30, timezone="UTC"),
        args=["openmeteo", 6],
        id="source-openmeteo-synoptic",
        name="live ingest openmeteo (00/06/12/18Z)",
        replace_existing=True,
        **_JOB_POLICY,
    )

    # ASOS obs + AFD — hourly (the daily-high label refines through the day; AFD pre-filtered, D-13).
    scheduler.add_job(
        _ingest_obs_all_cities,
        CronTrigger(minute=45, timezone="UTC"),
        id="obs-afd-hourly",
        name="live ingest asos obs + afd (hourly)",
        replace_existing=True,
        **_JOB_POLICY,
    )

    logger.info("built live scheduler with %d jobs (mode=live, D-15)", len(scheduler.get_jobs()))
    return scheduler


__all__ = ["build_scheduler"]
