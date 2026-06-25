"""Paired day-block bootstrap + Holm step-down (VER-05 / VER-06): CIs that respect day-pairing.

D-04 (verify subtree-local): the Gate-1 confidence intervals come from a PAIRED DAY-BLOCK
bootstrap — resample whole DAYS (with replacement), not individual contracts, so the per-day
correlation structure (many buckets share a day's weather outcome) is preserved and the CI is not
falsely narrowed. ``paired_day_block_ci`` is deterministic under a fixed ``seed``
(``np.random.default_rng``) so a verdict is reproducible. ``holm_step_down`` applies the Holm
step-down multiplicity correction across the secondary per-stratum tests (D-08 anti-p-hacking).

Pure NumPy + stdlib — no scipy/sklearn (AST-guarded). Bodies land Wave 3;
``tests/test_bootstrap.py`` + ``tests/test_secondaries.py`` pin the contracts (RED).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["paired_day_block_ci", "holm_step_down"]

# Bootstrap defaults: resamples, RNG seed, and the two-sided alpha (95% CI).
_DEFAULT_N_RESAMPLES = 10_000
_DEFAULT_SEED = 0
_DEFAULT_ALPHA = 0.05


def paired_day_block_ci(
    day_keys,
    score_fn,
    *,
    n_resamples: int = _DEFAULT_N_RESAMPLES,
    seed: int = _DEFAULT_SEED,
    alpha: float = _DEFAULT_ALPHA,
) -> tuple[float, float, NDArray[np.float64]]:
    """Paired day-block bootstrap CI for a paired score (VER-05).

    Resamples whole DAYS (``day_keys``) with replacement via ``np.random.default_rng(seed)``,
    re-evaluates ``score_fn`` per resample, and returns ``(ci_lo, ci_hi, resample_distribution)``
    at the ``1 - alpha`` level. Deterministic under ``seed`` (two calls → identical CI). Body
    lands Wave 3.
    """
    raise NotImplementedError("verify.bootstrap.paired_day_block_ci lands in Wave 3 (VER-05).")


def holm_step_down(pvalues, alpha: float = _DEFAULT_ALPHA) -> list[bool]:
    """Holm step-down multiplicity correction (VER-06).

    Returns a per-hypothesis reject/keep decision vector matching the textbook Holm procedure
    (sort ascending, compare the k-th smallest p-value to ``alpha / (m - k)``, stop at the first
    non-reject). Controls family-wise error across the secondary per-stratum tests (D-08). Body
    lands Wave 3.
    """
    raise NotImplementedError("verify.bootstrap.holm_step_down lands in Wave 3 (VER-06).")
