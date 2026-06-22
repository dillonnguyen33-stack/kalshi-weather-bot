"""Blended CDF → Kalshi bucket probabilities (D-04/D-05).

CDF differencing ``P(bucket) = Φ_blend(upper) − Φ_blend(lower)`` over the integer-°F ladder,
reusing the erf-based ``normal_cdf`` from :mod:`weatherquant.calibrate.crps` (D-04, Pitfall 6);
open buckets use the one-sided tail. Each integer degree ``k`` owns ``[k − _HALF, k + _HALF)``
(D-05, Pitfall 1). The exact label coverage is LOW-confidence; the live KXHIGH cross-check is
DEFERRED to Phase 5 (D-05; see docs/DECISIONS.md). Ticker/strike/label parsing lives in
:mod:`weatherquant.price.ticker`.

Pure NumPy + stdlib ``math`` only — no scipy/sklearn (AST guard).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from weatherquant.calibrate.crps import normal_cdf

__all__ = ["bucket_prob", "bucket_probs", "integers_in_bucket"]

# The single half-degree bucket-edge offset (D-05 / Pitfall 1): integer degree ``k`` owns
# ``[k − _HALF, k + _HALF)``. LOW-confidence; the live KXHIGH cross-check is DEFERRED to
# Phase 5 (see docs/DECISIONS.md) — do not treat as confirmed-against-live.
_HALF = 0.5


def integers_in_bucket(
    lo: int | None,
    hi: int | None,
    open_lo: bool = False,
    open_hi: bool = False,
) -> tuple[float, float]:
    """Continuous ``[lo − _HALF, hi + _HALF)`` span for a labeled bucket (D-05, Pitfall 1).

    Maps the inclusive integer degrees ``lo``..``hi`` to the single continuous interval for CDF
    differencing; contiguous integers collapse the per-integer spans to ``[lo − _HALF, hi + _HALF)``,
    which is what tiles a full ladder without gaps or overlaps. ``open_lo`` / ``open_hi`` mark
    tail buckets (``≤X`` / ``≥Y``): the open end is the ∓∞ sentinel, only the closed end offsets.
    """
    if open_lo:
        if hi is None:
            raise ValueError("integers_in_bucket: open_lo requires a finite upper integer hi.")
        return (-math.inf, float(hi) + _HALF)
    if open_hi:
        if lo is None:
            raise ValueError("integers_in_bucket: open_hi requires a finite lower integer lo.")
        return (float(lo) - _HALF, math.inf)

    if lo is None or hi is None:
        raise ValueError("integers_in_bucket: a closed bucket needs both integer edges lo, hi.")
    if hi < lo:
        raise ValueError(f"integers_in_bucket: inverted bucket (lo={lo} > hi={hi}).")
    return (float(lo) - _HALF, float(hi) + _HALF)


def _normal_cdf_scalar(z: float) -> float:
    """Scalar standard-normal CDF via the one promoted erf source of truth (D-04, Pitfall 6)."""
    return float(normal_cdf(np.array([z], dtype=np.float64))[0])


def bucket_prob(
    mu: float,
    sigma: float,
    lo: float,
    hi: float,
    open_lo: bool = False,
    open_hi: bool = False,
) -> float:
    """Probability mass in one continuous bucket by CDF differencing (D-04).

    ``Φ_blend(hi) − Φ_blend(lo)`` via the erf-based ``normal_cdf`` (Pitfall 6); ``lo``/``hi`` are
    CONTINUOUS edges already offset by ``±_HALF`` (see :func:`integers_in_bucket`). ``open_hi``
    collapses the upper edge to the ``1.0`` tail; ``open_lo`` the lower edge to ``0.0``. Fails
    loud (ASVS V5 / T-04-09): ``mu`` finite, ``sigma`` finite and > 0.
    """
    if not math.isfinite(mu):
        raise ValueError(f"bucket_prob: mu must be finite, got {mu!r}.")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"bucket_prob: sigma must be finite and > 0, got {sigma!r}.")
    if open_lo and open_hi:
        # A bucket open on BOTH ends would span (−∞, ∞) and silently return 1.0 — masking a
        # caller bug. The ticker parser only ever sets ONE open end, so fail loud (ASVS V5).
        raise ValueError(
            "bucket_prob: a bucket cannot be open on BOTH ends (would span all mass = 1.0)."
        )

    upper = 1.0 if open_hi else _normal_cdf_scalar((hi - mu) / sigma)
    lower = 0.0 if open_lo else _normal_cdf_scalar((lo - mu) / sigma)
    return upper - lower


def bucket_probs(
    mu: float,
    sigma: float,
    ladder: Sequence[tuple[float, float, bool, bool]],
) -> NDArray[np.float64]:
    """Probabilities across a full bucket ladder, summing to ~1 (D-04).

    ``ladder`` is a sequence of ``(lo, hi, open_lo, open_hi)`` continuous buckets tiling the
    line (open tails included); returns one float probability per bucket. A gapless tiling of
    ``(−∞, ∞)`` sums to ~1. ``mu``/``sigma`` are guarded per bucket by :func:`bucket_prob`.
    """
    return np.array(
        [bucket_prob(mu, sigma, lo, hi, open_lo, open_hi) for lo, hi, open_lo, open_hi in ladder],
        dtype=np.float64,
    )
