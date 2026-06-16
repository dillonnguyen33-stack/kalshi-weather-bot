"""AsyncIOScheduler wiring (D-15) — the LIVE half of the one ingestion code path.

``build_scheduler`` returns a configured (NOT-yet-started) ``AsyncIOScheduler`` whose jobs
call the SAME :func:`weatherquant.ingest.orchestrator.ingest_cycle` the backfill CLI calls
— the only difference is ``mode="live"`` (so ``available_at`` is the decode-completion
``now(UTC)``, D-09) versus the CLI's ``mode="backfill"``. There is no second, drifting live
ingestion path the Phase-6 backtest could diverge from (D-15).

APSCHEDULER PIN (T-02-SC). This uses ``apscheduler.schedulers.asyncio.AsyncIOScheduler``
from apscheduler **3.11.x** — never the in-flux 4.x rewrite (whose scheduler package and
import paths differ entirely; pinned away in pyproject and proven by the scheduler unit
test). ``AsyncIOScheduler`` shares the asyncio event loop with the httpx sources and
(future) the Kalshi WebSocket feed.

PER-MODEL CADENCE (RESEARCH Pattern 8). NWP cycles are deterministic:

* HRRR / NBM — hourly (``CronTrigger(minute=...)``).
* GFS / GEFS — the 00/06/12/18Z synoptic cycles (``CronTrigger(hour="0,6,12,18")``).
* NWS gridpoint + Open-Meteo ensemble + ASOS obs + AFD — sensible supplementary cadences
  (hourly NWS/obs, per-GFS-cycle Open-Meteo, hourly-ish AFD) without over-engineering.

``build_scheduler`` does NOT start the scheduler at import or build time — it returns the
configured object so it is unit-testable (``get_jobs()`` asserts the cadence) and the caller
owns the lifecycle (``scheduler.start()`` inside a running loop, then ``asyncio.Event().wait()``).
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

# Cycle latency cushion: when a cadence fires we ingest the cycle that has had time to
# publish (cycle_init = the firing hour). The orchestrator's per-source try/except handles a
# not-yet-published cycle as a graceful fallback (D-11), so a small clock skew never crashes.


def _latest_synoptic_cycle(now: datetime, step_hours: int) -> datetime:
    """Floor ``now`` to the most recent ``step_hours`` synoptic cycle (e.g. 6h -> 00/06/12/18Z)."""
    floored_hour = (now.hour // step_hours) * step_hours
    return now.replace(hour=floored_hour, minute=0, second=0, microsecond=0)


def _latest_hourly_cycle(now: datetime) -> datetime:
    """Floor ``now`` to the top of the current hour (the latest HRRR/NBM cycle)."""
    return now.replace(minute=0, second=0, microsecond=0)


async def _ingest_grib_all_cities(model: str, step_hours: int) -> None:
    """Live job body: ingest ``model`` for the latest cycle across EVERY registry city (D-15).

    Calls the SAME :func:`orchestrator.ingest_cycle` the CLI backfill uses, with
    ``mode="live"`` (D-15/D-09). Each city is independent — the orchestrator's graceful
    degradation means one city's missing cycle never blocks the others (D-11).
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

    Registers one job per model cadence, each calling the SAME orchestrator function the CLI
    backfill calls (``mode="live"``):

    * ``hrrr`` / ``nbm`` — hourly ``CronTrigger`` (top of every hour).
    * ``gfs`` / ``gefs`` — ``CronTrigger(hour="0,6,12,18")`` (the synoptic cycles).
    * ``nws`` — hourly; ``openmeteo`` — per-GFS-cycle; obs/AFD — hourly.

    The scheduler is returned UNSTARTED so it is unit-testable (``get_jobs()``) and the caller
    owns ``start()``. Returns the configured :class:`AsyncIOScheduler`.
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

    # NWS gridpoint — hourly; Open-Meteo ensemble — per-GFS cycle (sensible, not over-engineered).
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

    # ASOS obs + AFD — hourly (the daily-high label refines through the day; AFD is cheap
    # thanks to the keyword pre-filter, D-13).
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
