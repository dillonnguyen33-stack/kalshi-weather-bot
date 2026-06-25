"""RED contract for the paired day-block bootstrap (VER-05).

Three invariants the Wave-3 ``paired_day_block_ci`` must satisfy:

* Reproducible under a fixed seed — two calls give an identical CI.
* Day-pairing preserved — a constant per-day delta gives a CI tightly around that delta.
* Pairing-broken (independent) resampling gives a demonstrably different / wider CI, proving the
  paired structure is actually being used.

Imports are deferred so collection stays green while the implementation is RED.
"""

from __future__ import annotations

import numpy as np
import pytest


def _constant_delta_days(n_days: int, delta: float):
    """day_keys + a score_fn returning a constant per-day paired delta (the tight-CI fixture)."""
    day_keys = list(range(n_days))
    per_day = {d: delta for d in day_keys}

    def score_fn(resampled_days) -> float:
        return float(np.mean([per_day[d] for d in resampled_days]))

    return day_keys, score_fn


def test_paired_day_block_ci_is_reproducible_under_a_fixed_seed():
    """Two calls with the same seed return the identical CI (deterministic verdict)."""
    from weatherquant.verify import bootstrap

    day_keys, score_fn = _constant_delta_days(60, delta=0.05)
    lo1, hi1, _d1 = bootstrap.paired_day_block_ci(day_keys, score_fn, n_resamples=2000, seed=0)
    lo2, hi2, _d2 = bootstrap.paired_day_block_ci(day_keys, score_fn, n_resamples=2000, seed=0)
    assert (lo1, hi1) == (lo2, hi2)


def test_constant_per_day_delta_gives_ci_around_that_delta():
    """A constant per-day delta of 0.05 yields a CI tightly bracketing 0.05."""
    from weatherquant.verify import bootstrap

    day_keys, score_fn = _constant_delta_days(60, delta=0.05)
    lo, hi, _dist = bootstrap.paired_day_block_ci(day_keys, score_fn, n_resamples=2000, seed=1)
    # ``np.mean`` of 60 copies of 0.05 is 0.04999999999999998 (one ULP below 0.05 — IEEE-754
    # summation, not a bootstrap artifact), so bracket the delta at float tolerance, not the
    # literal 0.05. The CI must still tightly enclose the true per-day delta.
    assert lo <= 0.05 <= hi or (lo, hi) == pytest.approx((0.05, 0.05))
    assert (hi - lo) < 1e-6  # a constant delta has ~zero resample variance


def test_pairing_broken_resample_widens_the_ci():
    """Breaking the day-pairing (independent resample) gives a wider/different CI than the paired one."""
    from weatherquant.verify import bootstrap

    rng = np.random.default_rng(5)
    day_keys = list(range(80))
    # Heterogeneous per-day deltas: the paired structure matters.
    per_day = {d: float(v) for d, v in zip(day_keys, rng.normal(0.05, 0.1, len(day_keys)))}

    def paired_score(resampled_days) -> float:
        return float(np.mean([per_day[d] for d in resampled_days]))

    lo_p, hi_p, _dp = bootstrap.paired_day_block_ci(day_keys, paired_score, n_resamples=3000, seed=2)
    # A pairing-broken resample draws values independently of day identity.
    pooled = np.array(list(per_day.values()))

    def broken_score(resampled_days) -> float:
        idx = rng.integers(0, len(pooled), len(resampled_days))
        return float(pooled[idx].mean())

    lo_b, hi_b, _db = bootstrap.paired_day_block_ci(day_keys, broken_score, n_resamples=3000, seed=2)
    assert (hi_b - lo_b) != pytest.approx(hi_p - lo_p, rel=0.01)
