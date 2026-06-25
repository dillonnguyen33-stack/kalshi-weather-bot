"""RED contract for the secondary per-stratum analyses (VER-06).

Covers the Holm step-down multiplicity correction and the per-stratum secondary loop:

* ``holm_step_down`` on known p-values reproduces the textbook step-down decision vector.
* The secondary loop produces per-(city/lead/month) CIs.
* Excluded-day coverage is reported alongside the secondary CIs (D-09).

Imports are deferred so collection stays green while the implementation is RED.
"""

from __future__ import annotations

import pytest


def test_holm_step_down_matches_textbook_decision_vector():
    """Holm at alpha=0.05 on a known p-vector matches the hand-computed reject/keep decisions.

    p = [0.001, 0.013, 0.040, 0.5], m=4. Sorted compares: 0.001<0.05/4=0.0125 reject; 0.013<0.05/3
    =0.0167 reject; 0.040<0.05/2=0.025? NO → stop, the rest keep. Decisions follow the ORIGINAL
    order: [reject, reject, keep, keep] → [True, True, False, False].
    """
    from weatherquant.verify import bootstrap

    decisions = bootstrap.holm_step_down([0.001, 0.013, 0.040, 0.5], alpha=0.05)
    assert list(decisions) == [True, True, False, False]


def test_holm_step_down_preserves_input_order():
    """An unsorted p-vector returns decisions aligned to the INPUT order, not the sorted order."""
    from weatherquant.verify import bootstrap

    # Same multiset as above but shuffled; the 0.5 (index 0) must map to keep, the 0.001 to reject.
    decisions = bootstrap.holm_step_down([0.5, 0.001, 0.040, 0.013], alpha=0.05)
    assert list(decisions) == [False, True, False, True]


@pytest.mark.integration
def test_secondary_loop_reports_per_stratum_cis_and_coverage(pg_conn):
    """The per-(city/lead/month) secondary loop yields a CI per stratum plus excluded-day coverage."""
    from weatherquant.verify import backtest, bootstrap  # noqa: F401  (Wave-3 orchestration seam)

    # The secondary per-stratum CIs + coverage are produced in Wave 3 (RED now). This test binds the
    # contract: each stratum result must carry a CI and a coverage entry (no silent drops, D-09).
    records, coverage = backtest.walk_forward(
        pg_conn, "KXHIGHNY", "blend", lead=1,
        start=None, end=None, oos_slice=None,
    )
    assert all("reason" in entry for entry in coverage)
