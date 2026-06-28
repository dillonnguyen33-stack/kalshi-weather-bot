"""``weatherquant ingest`` — backfill via the orchestrator (mode=backfill, D-09/D-10/D-15).

``get_engine`` and ``orchestrator`` are imported into this module's namespace so the run body
resolves ``cli.ingest.get_engine`` / ``cli.ingest.orchestrator`` (the seams tests monkeypatch).
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from weatherquant.db.engine import get_engine
from weatherquant.ingest import orchestrator

from ._args import (
    _resolve_cities,
    _resolve_cycle_hours,
    _resolve_models,
    _resolve_range,
)

logger = logging.getLogger(__name__)


def run_ingest(args: argparse.Namespace) -> dict[str, int]:
    """Validate args and dispatch to the orchestrator backfill path (mode=backfill, D-15).

    Cities and dates are validated (the argparse ``type=`` validators already rejected
    unknown cities and malformed dates); the engine bind is built from the validated
    ``DATABASE_URL`` and the async :func:`ingest_range` is driven via ``asyncio.run``.
    Re-running the same range is a no-op (idempotency, D-10).
    """
    start, end = _resolve_range(args)
    models = _resolve_models(args)
    cities = _resolve_cities(args)
    cycle_hours = _resolve_cycle_hours(args)

    bind = get_engine()
    logger.info(
        "ingest backfill models=%s cities=%s range=%s..%s lead=%s cycle_hours=%s",
        models,
        cities,
        start,
        end,
        args.lead,
        cycle_hours,
    )
    totals = asyncio.run(
        orchestrator.ingest_range(
            bind,
            models,
            cities,
            start,
            end,
            mode="backfill",
            lead=args.lead,
            cycle_hours=cycle_hours,
        )
    )
    logger.info("ingest complete: rows inserted per model=%s", totals)
    return totals
