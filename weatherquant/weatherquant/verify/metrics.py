"""Gate-1 metric core (VER-01): Brier (+ Murphy decomposition), ECE, PIT, CRPS, ROI, CLV.

D-07 (verify subtree-local): the proof metrics are hand-rolled NumPy + stdlib — no scipy/sklearn
(fenced by ``tests/test_no_forbidden_verify_deps.py``). The Gaussian CRPS and the erf-based normal
CDF are REUSED from :mod:`weatherquant.calibrate.crps` (the one source of truth, D-04) — never
re-derived here — so ``crps_blend`` agrees with the calibration core by construction.

Bodies land in Wave 2 — every function raises ``NotImplementedError``; the contracts are pinned by
``tests/test_metrics.py`` (RED until then).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Imported now (D-04 single source of truth) so the Wave-2 bodies reuse the one closed-form CRPS /
# erf-based CDF rather than re-deriving them; referenced once the stubs land.
from weatherquant.calibrate.crps import crps_norm, normal_cdf  # noqa: F401  (Wave-2 seam)

__all__ = [
    "brier",
    "brier_murphy",
    "ece_equal_count",
    "pit_values",
    "crps_blend",
    "roi_from_fills",
    "mean_clv",
]

# Default bin count for the Murphy decomposition / equal-count ECE (overridable per call).
_DEFAULT_N_BINS = 10


def brier(f: NDArray[np.float64], o: NDArray[np.float64]) -> float:
    """Mean Brier score ``mean((f - o)^2)`` over forecast probabilities ``f`` and outcomes ``o``.

    ``f`` are predicted YES probabilities in ``[0, 1]``; ``o`` are realized ``{0, 1}`` outcomes.
    Lower is better. Body lands Wave 2.
    """
    raise NotImplementedError("verify.metrics.brier lands in Wave 2 (VER-01).")


def brier_murphy(
    f: NDArray[np.float64], o: NDArray[np.float64], n_bins: int = _DEFAULT_N_BINS
) -> dict[str, float]:
    """Murphy 3-component decomposition of the Brier score (VER-01).

    Returns ``{"reliability", "resolution", "uncertainty"}`` such that the binned Brier equals
    ``reliability - resolution + uncertainty``. ``n_bins`` equal-WIDTH bins over ``f``. Body
    lands Wave 2.
    """
    raise NotImplementedError("verify.metrics.brier_murphy lands in Wave 2 (VER-01).")


def ece_equal_count(
    f: NDArray[np.float64], o: NDArray[np.float64], n_bins: int = _DEFAULT_N_BINS
) -> float:
    """Expected Calibration Error over ``n_bins`` EQUAL-COUNT (quantile) bins (VER-01).

    Equal-count bins (not equal-width) so each bin carries comparable sample mass; ECE is the
    sample-weighted mean ``|mean(f) - mean(o)|`` per bin. ~0 for a perfectly calibrated forecast,
    > 0 for a biased one. Body lands Wave 2.
    """
    raise NotImplementedError("verify.metrics.ece_equal_count lands in Wave 2 (VER-01).")


def pit_values(
    y: NDArray[np.float64], mu: NDArray[np.float64], sigma: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Probability Integral Transform values ``Phi((y - mu)/sigma)`` (VER-01).

    Reuses the erf-based ``normal_cdf`` (D-04). For a correctly calibrated Gaussian forecast the
    PIT values are ~Uniform(0, 1). Body lands Wave 2.
    """
    raise NotImplementedError("verify.metrics.pit_values lands in Wave 2 (VER-01).")


def crps_blend(
    mu: NDArray[np.float64], sigma: NDArray[np.float64], y: NDArray[np.float64]
) -> float:
    """Mean Gaussian CRPS of the blended predictive against the verifying obs (VER-01).

    Delegates elementwise to :func:`weatherquant.calibrate.crps.crps_norm` (the one closed-form
    source, D-04) and returns the mean — never a re-derived CRPS. Body lands Wave 2.
    """
    raise NotImplementedError("verify.metrics.crps_blend lands in Wave 2 (VER-01).")


def roi_from_fills(fills, settled_yes) -> float:
    """Realized return-on-investment over paper fills given per-market YES settlement (VER-01).

    ROI = net P&L / capital deployed across the paired Gate-1 fills, fee-aware. Body lands Wave 2.
    """
    raise NotImplementedError("verify.metrics.roi_from_fills lands in Wave 2 (VER-01).")


def mean_clv(fills, closing_snapshots, sides) -> float:
    """Mean Closing-Line Value (cents) over the paper fills vs the closing volume-weighted mid.

    Reuses the Phase-5 derived-CLV convention (closing mid − fill price, sign by side); positive
    mean CLV is the edge-vs-close signal. Body lands Wave 2 (VER-01).
    """
    raise NotImplementedError("verify.metrics.mean_clv lands in Wave 2 (VER-01).")
