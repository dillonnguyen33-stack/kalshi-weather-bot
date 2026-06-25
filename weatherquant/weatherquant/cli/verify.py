"""``weatherquant verify`` — run the Gate-1 paired proof + render the verdict (D-12 / VER-* / SYS-02).

D-12 (verify subtree-local; CLI subcommand): ``run_verify`` is the terminal Gate-1 entry point. It
orchestrates the walk-forward paired backtest → paired day-block bootstrap CIs → pre-registered
conjunctive gate verdict → report render, and (for the drift subcommand) propagates a non-zero exit
code on a calibration breach (SYS-02).

``get_engine`` / ``get_settings`` are imported INTO this module's namespace (the test seam — the run
body resolves ``cli.verify.get_engine`` / ``cli.verify.get_settings``); the heavy
``weatherquant.verify`` imports stay LAZY in the body so ``--help`` and the ingest path never pay for
matplotlib or the metric core. ``run_verify`` returns an ``int`` exit code so a drift breach can
propagate non-zero (SYS-02). Body lands Waves 3-4; the import surface exists now.
"""

from __future__ import annotations

import argparse
import logging

# Imported INTO the module namespace as the test seam (the run body resolves
# ``cli.verify.get_engine`` / ``cli.verify.get_settings``); wired up once the body lands Waves 3-4.
from weatherquant.db.engine import get_engine, get_settings  # noqa: F401  (Waves 3-4 seam)

logger = logging.getLogger(__name__)


def run_verify(args: argparse.Namespace) -> int:
    """Run the Gate-1 paired proof / drift monitor and return a process exit code (D-12).

    Resolves the engine/settings via the namespace seams, then (lazily) drives
    ``verify.backtest`` → ``verify.bootstrap`` → ``verify.gate1`` → ``verify.report`` for the proof,
    or ``verify.drift`` for the monitor. Returns ``0`` on a clean pass and a NON-ZERO code on a
    Gate-1 FAIL or a drift breach (SYS-02). Body lands Waves 3-4.
    """
    # Imported lazily so `--help` and the non-verify paths never pay the metric/report imports.
    from weatherquant.verify import (  # noqa: F401  (import surface; bodies land Waves 3-4)
        backtest,
        bootstrap,
        drift,
        gate1,
        metrics,
        report,
    )

    raise NotImplementedError("cli.verify.run_verify lands in Waves 3-4 (D-12 / VER-* / SYS-02).")
