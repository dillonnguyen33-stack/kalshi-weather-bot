"""The ``weatherquant`` CLI — four subcommands over the shared spine (stdlib argparse).

* ``ingest``    — backfill via :func:`ingest.orchestrator.ingest_range` (``mode="backfill"``,
  so ``available_at = cycle_init + PUBLISH_LATENCY``, never the wall clock; idempotent re-run
  is a no-op). Same orchestrator the live scheduler calls, differing only in ``mode``
  (D-09/D-10/D-15).
* ``calibrate`` — fit + persist EMOS/NGR params per stratum.
* ``price``     — latest → predict → blend → bucket/EV/Kelly against a mocked mid (D-16).
* ``paper``     — paper-only book loop: snapshot + sized position to settlement (D-03/D-04/D-08).

Installed as the ``weatherquant`` console script via ``[project.scripts]``. This package re-exports
the pre-split public surface so ``weatherquant.cli:main`` and ``from weatherquant.cli import …``
keep resolving unchanged after the module→package split.
"""

from __future__ import annotations

from ._args import ALL_MODELS, build_parser
from .calibrate import run_calibrate
from .ingest import run_ingest
from .main import main
from .paper import PAPER_SNAPSHOT_CADENCE_SECONDS, run_paper
from .pricing import run_price
from .verify import run_verify

__all__ = [
    "ALL_MODELS",
    "PAPER_SNAPSHOT_CADENCE_SECONDS",
    "build_parser",
    "main",
    "run_calibrate",
    "run_ingest",
    "run_paper",
    "run_price",
    "run_verify",
]
