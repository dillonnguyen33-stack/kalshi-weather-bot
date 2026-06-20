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

import logging
from datetime import datetime, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from weatherquant.db.engine import get_engine
from weatherquant.ingest import orchestrator
from weatherquant.registry import CITIES

logger = logging.getLogger(__name__)

# Cycle latency cushion: a cadence fires for the cycle that has had time to publish; the
# orchestrator handles a not-yet-published cycle gracefully (D-11), so clock skew never crashes.


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
    now = datetime.now(timezone.utc)
    cycle = (
        _latest_hourly_cycle(now) if step_hours == 1 else _latest_synoptic_cycle(now, step_hours)
    )
    bind = get_engine()
    for city in CITIES:
        await orchestrator.ingest_cycle(bind, model, city, cycle, mode="live", lead=0)
    logger.info("live grib job model=%s cycle=%s ran across %d cities", model, cycle, len(CITIES))


async def _ingest_source_all_cities(source: str, step_hours: int) -> None:
    """Live job body for a supplementary source (nws/openmeteo) across every city (D-15)."""
    now = datetime.now(timezone.utc)
    cycle = (
        _latest_hourly_cycle(now) if step_hours == 1 else _latest_synoptic_cycle(now, step_hours)
    )
    bind = get_engine()
    for city in CITIES:
        await orchestrator.ingest_cycle(bind, source, city, cycle, mode="live", lead=0)
    logger.info("live source job source=%s cycle=%s ran across %d cities", source, cycle, len(CITIES))


async def _ingest_obs_all_cities() -> None:
    """Live job body: ASOS daily-high + AFD signal for today across every city (02-03)."""
    today = datetime.now(timezone.utc).date()
    bind = get_engine()
    for city in CITIES:
        await orchestrator.ingest_obs(bind, city, today)
        await orchestrator.ingest_afd(bind, city, today)
    logger.info("live obs/afd job ran for %s across %d cities", today, len(CITIES))


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
        )

    # NWS gridpoint — hourly; Open-Meteo ensemble — per-GFS cycle.
    scheduler.add_job(
        _ingest_source_all_cities,
        CronTrigger(minute=15, timezone="UTC"),
        args=["nws", 1],
        id="source-nws-hourly",
        name="live ingest nws (hourly)",
        replace_existing=True,
    )
    scheduler.add_job(
        _ingest_source_all_cities,
        CronTrigger(hour="0,6,12,18", minute=30, timezone="UTC"),
        args=["openmeteo", 6],
        id="source-openmeteo-synoptic",
        name="live ingest openmeteo (00/06/12/18Z)",
        replace_existing=True,
    )

    # ASOS obs + AFD — hourly (the daily-high label refines through the day; AFD pre-filtered, D-13).
    scheduler.add_job(
        _ingest_obs_all_cities,
        CronTrigger(minute=45, timezone="UTC"),
        id="obs-afd-hourly",
        name="live ingest asos obs + afd (hourly)",
        replace_existing=True,
    )

    logger.info("built live scheduler with %d jobs (mode=live, D-15)", len(scheduler.get_jobs()))
    return scheduler


__all__ = ["build_scheduler"]
