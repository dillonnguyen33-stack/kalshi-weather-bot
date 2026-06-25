"""Paired day-block bootstrap + Holm step-down (VER-05 / VER-06): CIs that respect day-pairing.

D-04 (verify subtree-local): the Gate-1 confidence intervals come from a PAIRED DAY-BLOCK
bootstrap — resample whole DAYS (with replacement), not individual contracts, so the per-day
correlation structure (many buckets share a day's weather outcome) is preserved and the CI is not
falsely narrowed. ``paired_day_block_ci`` is deterministic under a fixed ``seed``
(``np.random.default_rng``) so a verdict is reproducible. ``holm_step_down`` applies the Holm
step-down multiplicity correction across the secondary per-stratum tests (D-05 anti-p-hacking).

Block = calendar DAY (we resample day-keys with replacement directly), NOT a moving/circular
block over a time-ordered metric series. This is the locked D-08 reading: the day is the
independence unit, each resampled day carries BOTH arms' raw per-market records (so the delta is
always paired on the same markets), and resampling day labels is the simplest design that
preserves pairing + serial structure. The moving-block bootstrap (Künsch 1989) is the reference
alternative for intra-window autocorrelation beyond the day — it is explicitly NOT chosen here and
must not be substituted (it would change the pre-registered CI width, D-08).

Holm (``holm_step_down``) applies ONLY to the secondary per-stratum comparisons. The five PRIMARY
Gate-1 metrics are NOT Holm-adjusted: D-13 is a *conjunctive* rule (ALL five CIs must individually
exclude zero), and a conjunction is conservative by construction — requiring more tests to all
pass makes the gate harder, not easier, so there is no multiplicity inflation to correct (RESEARCH
Pattern 7/8). Adjusting the primaries would be a category error.

Pure NumPy + stdlib — no scipy/sklearn (AST-guarded).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

__all__ = ["paired_day_block_ci", "holm_step_down"]

# Bootstrap defaults: resamples, RNG seed, and the two-sided alpha (95% CI).
_DEFAULT_N_RESAMPLES = 10_000
_DEFAULT_SEED = 0
_DEFAULT_ALPHA = 0.05


def paired_day_block_ci(
    day_keys: Sequence,
    score_fn: Callable[[NDArray], float],
    *,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    seed: int = _DEFAULT_SEED,
    alpha: float = _DEFAULT_ALPHA,
) -> tuple[float, float, NDArray[np.float64]]:
    """Paired day-block bootstrap CI for a paired score (VER-05, D-04).

    Resamples ``|unique-days|`` day-keys WITH REPLACEMENT (block = one calendar day) via
    ``np.random.default_rng(seed)``, calls ``score_fn(sampled_days) -> float`` (the pooled
    ``wq - v3`` delta over those days), repeats ``n_resamples`` times, and returns
    ``(ci_lo, ci_hi, deltas)`` at the two-sided ``1 - alpha`` percentiles (2.5 / 97.5 for the
    default 95% CI).

    Pairing is preserved by the caller's ``score_fn``: each resampled day carries BOTH arms' raw
    per-market records (the score pools raw records, never pre-averaged per-day deltas), so the
    delta is always computed on the same markets. Block = day, so the within-day serial-correlation
    structure (many buckets share a day's weather outcome) is respected and the CI is not falsely
    narrowed (RESEARCH §Pitfall 2).

    Deterministic under ``seed``: two calls with the same seed return a byte-identical CI, so the
    Gate-1 verdict is reproducible (the seed is recorded in the pre-registration / verdict).

    Args:
        day_keys: the settlement-day keys in the test window (duplicates are de-duplicated;
            ``|unique-days|`` keys are resampled per replicate).
        score_fn: ``score_fn(sampled_day_keys) -> float`` returning the pooled paired delta for
            those resampled days.
        n_resamples: bootstrap replicate count ``R`` (default 10,000).
        seed: RNG seed for ``np.random.default_rng`` (reproducibility — same seed → same CI).
        alpha: two-sided miscoverage (default 0.05 → 95% CI).

    Returns:
        ``(ci_lo, ci_hi, deltas)`` — the percentile CI bounds and the full resample distribution.
    """
    rng = np.random.default_rng(seed)
    uniq = np.asarray(sorted(set(day_keys)))
    deltas = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        sampled = rng.choice(uniq, size=uniq.size, replace=True)
        deltas[r] = score_fn(sampled)
    lo, hi = np.percentile(deltas, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi), deltas


def holm_step_down(pvalues: Sequence[float], alpha: float = _DEFAULT_ALPHA) -> list[bool]:
    """Holm-Bonferroni step-down multiplicity correction for the SECONDARIES only (VER-06, D-05).

    Controls the family-wise error rate across the ``m`` secondary per-(city/lead/month)
    comparisons. Standard step-down procedure:

    1. sort the p-values ascending: ``p_(1) <= p_(2) <= ... <= p_(m)``;
    2. for the ``k``-th smallest (1-indexed), compare ``p_(k)`` to ``alpha / (m - k + 1)``;
    3. reject while ``p_(k) <= alpha / (m - k + 1)``, and STOP at the first failure — every
       subsequent (larger) hypothesis is NOT rejected (the step-down property).

    Returns the per-input reject mask in the ORIGINAL input order.

    This applies ONLY to the descriptive secondary per-stratum comparisons. The five PRIMARY
    Gate-1 metrics are NOT Holm-adjusted — D-13 is a conjunctive AND (all five CIs must exclude
    zero), which is conservative by construction and is not a multiplicity-inflation problem
    (RESEARCH Pattern 7/8).

    Args:
        pvalues: the secondary two-sided p-values (e.g. ``2*min(mean(d>=0), mean(d<=0))`` clipped
            to ``[1/R, 1]`` from the bootstrap).
        alpha: family-wise error rate (default 0.05).

    Returns:
        A ``list[bool]`` reject mask aligned to the input order (``True`` = reject / significant).
    """
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    reject = [False] * m
    if m == 0:
        return reject
    order = np.argsort(p, kind="stable")  # ascending; stable so ties keep input order
    for k, idx in enumerate(order):  # k is 0-indexed → threshold uses (m - k)
        threshold = alpha / (m - k)
        if p[idx] <= threshold:
            reject[int(idx)] = True
        else:
            break  # step-down: stop at the first non-reject; the rest stay False
    return reject
