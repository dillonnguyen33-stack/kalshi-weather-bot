"""Gate-1 report rendering (VER-02 / D-11): reliability + PIT PNGs and the GATE1-VERDICT artifacts.

D-11 (verify subtree-local; matplotlib isolation): this is the SOLE module in the verify subtree
that may import matplotlib — and the import is kept LAZY inside :func:`render_reports` so the
pure-NumPy metric core (``metrics``/``bootstrap``/``gate1``) never pays for it and the no-forbidden
AST guard fences scipy/sklearn out of the core while explicitly EXCLUDING this reporting edge.

``render_reports`` writes per-stratum reliability + PIT-histogram PNGs and the
``GATE1-VERDICT.{json,md}`` (the pooled CIs, PASS/FAIL, the RNG seed, and excluded-day coverage)
into a project-relative, gitignored ``reports/`` dir. Output filenames are built from VALIDATED city
codes only (V12 path-safety) — no untrusted path segments. Body lands Wave 4;
``tests/test_report.py`` pins the contract (RED).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["render_reports"]


def render_reports(records, cis, verdict, *, out_dir) -> dict[str, str]:
    """Render the Gate-1 reliability/PIT PNGs + GATE1-VERDICT.{json,md} into ``out_dir`` (VER-02).

    matplotlib is imported LAZILY inside this body (D-11 isolation) — never at module top. Writes
    one reliability + one PIT PNG per stratum and the verdict JSON/MD (pooled CIs, PASS/FAIL, RNG
    seed, excluded-day coverage); returns a mapping of artifact name → written path. Filenames are
    built from validated city codes only (V12 path-safety). Body lands Wave 4.
    """
    raise NotImplementedError("verify.report.render_reports lands in Wave 4 (VER-02).")
