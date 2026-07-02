"""``weatherquant live`` — start the live ingestion scheduler (D-15) and block until interrupted.

``scheduler.build_scheduler`` is deliberately returned UNSTARTED so it stays unit-testable; this
command is the caller that owns the asyncio loop. It starts the scheduler, runs forever, and on
SIGINT shuts it down cleanly. This is the continuous half of the one ingestion code path — the
forecasts/obs/AFD that calibration and paper-trading read accumulate while this runs.
"""

from __future__ import annotations

import asyncio
import logging

from weatherquant.scheduler import build_scheduler

logger = logging.getLogger(__name__)


async def _serve() -> None:
    """Start the scheduler and await cancellation (SIGINT), shutting it down in the finally."""
    scheduler = build_scheduler()
    scheduler.start()
    logger.info("live scheduler started (%d jobs); Ctrl-C to stop", len(scheduler.get_jobs()))
    try:
        await asyncio.Event().wait()  # never set — run until the loop is interrupted.
    finally:
        scheduler.shutdown(wait=False)
        logger.info("live scheduler stopped")


def run_live(_args: object) -> int:
    """Run the live scheduler until interrupted; returns 0 on a clean stop (Ctrl-C)."""
    try:
        asyncio.run(_serve())
    except KeyboardInterrupt:
        pass
    return 0


__all__ = ["run_live"]
