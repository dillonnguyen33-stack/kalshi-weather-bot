"""Walk-forward paired backtest (VER-03 / D-09 / D-12): as-of-correct WQ-vs-v3 record assembly.

D-12 (verify subtree-local): ``walk_forward`` replays the ledger day-by-day, building one
``PairedRecord`` per (day, city, bucket) that scores the Weatherquant blended probability against
the legacy v3 probability on the SAME realized outcome. It is strictly as-of-correct: only ledger
rows with ``available_at < cutoff`` are consumed (no look-ahead), and the Gate-1 test window is
asserted DISJOINT from the Phase-3 OOS slice so the proof is never scored on tuning data.

D-09 (absence is absence): voided / missing / non-settling days are NOT silently dropped — each
appears in the returned coverage log with an explicit ``excluded_reason`` so excluded-day coverage
is auditable in the final verdict.

Bodies land Wave 3 — ``walk_forward`` raises ``NotImplementedError``; ``PairedRecord`` is a frozen
dataclass usable now. Contracts pinned by ``tests/test_backtest_no_lookahead.py`` (RED).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = ["PairedRecord", "walk_forward"]


@dataclass(frozen=True)
class PairedRecord:
    """One as-of-correct paired observation: WQ vs v3 on the same (day, city, bucket) outcome.

    ``o_i`` is the realized ``{0, 1}`` YES outcome for the bucket. ``excluded_reason`` is ``None``
    for a scored row, or a short reason string for a coverage-logged exclusion (D-09 — never a
    silent drop).
    """

    day: object
    city: str
    bucket: object
    wq_prob: float
    v3_prob: float
    o_i: int
    excluded_reason: str | None = None


def walk_forward(bind, city, model, lead, start, end, oos_slice):
    """Replay the ledger as-of-correctly and assemble paired WQ-vs-v3 records (VER-03).

    Consumes ONLY rows with ``available_at < cutoff`` (no look-ahead, D-12), asserts the
    ``[start, end)`` test window is disjoint from ``oos_slice`` (the Phase-3 OOS span), and
    coverage-logs every voided/missing day with a reason (D-09). Returns
    ``(records, coverage_log)``. Body lands Wave 3.
    """
    raise NotImplementedError("verify.backtest.walk_forward lands in Wave 3 (VER-03).")
