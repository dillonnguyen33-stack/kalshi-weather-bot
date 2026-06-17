"""Blended CDF → Kalshi bucket probabilities + ticker parser (PRC-02, D-04/D-05/D-06).

Map the continuous blended Gaussian onto the integer-°F Kalshi bucket ladder by CDF
differencing — ``P(bucket) = Φ_blend(upper) − Φ_blend(lower)`` — reusing the erf-based
normal CDF promoted public in :mod:`weatherquant.calibrate.crps` (``normal_cdf``), never a
second erf implementation (D-04, RESEARCH Pitfall 6). Open buckets (``≤X``, ``≥Y``) use the
one-sided tail. A full ladder's probabilities are asserted to sum to ~1 by
``tests/test_buckets.py -k sum``.

The settled value is a whole-°F NWS Daily Climate Report high, but the predictive
distribution is continuous, so each integer degree ``k`` owns the half-open continuous
interval ``[k − _HALF, k + _HALF)`` and a bucket's mass sums ``Φ(k+_HALF) − Φ(k−_HALF)`` over
the integers it covers (D-05, RESEARCH Pitfall 1). The half-degree offset lives in ONE place
(``_HALF``) and the inclusive-integer coverage in ONE helper (``integers_in_bucket``); the
exact coverage of a label is LOW-confidence and is gated behind a ``checkpoint:human-verify``
against a live ``KXHIGH`` market before the offset is locked.

``parse_ticker`` is a pure string→edges parser (no I/O, D-06): it fails loud on a malformed
ticker (raise, never silently default an edge — ASVS V5) and prefers the structured
``floor_strike``/``cap_strike`` the Kalshi API supplies over label parsing.

Pure NumPy + stdlib ``math`` only — no scipy/sklearn (the AST guard enforces it).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from weatherquant.calibrate.crps import normal_cdf

__all__ = ["integers_in_bucket", "bucket_prob", "bucket_probs", "parse_ticker"]

# The single half-degree bucket-edge offset (D-05 / RESEARCH Pitfall 1): integer degree ``k``
# owns the continuous interval ``[k − _HALF, k + _HALF)``. Centralized here so a one-place
# change (after the live-market human-verify checkpoint) re-maps every bucket consistently.
# LOW-confidence value: the exact inclusive-integer coverage of a label is locked only by the
# 04-06 ``checkpoint:human-verify`` against a live ``KXHIGH`` market — do not treat it as final.
_HALF = 0.5


def integers_in_bucket(
    lo: int | None,
    hi: int | None,
    open_lo: bool = False,
    open_hi: bool = False,
) -> tuple[float, float]:
    """Continuous ``[k − _HALF, k + _HALF)`` span for a labeled bucket (D-05, Pitfall 1).

    Maps the inclusive integer degrees a bucket label covers (``lo``..``hi``) to the single
    continuous interval used for CDF differencing: the lowest integer ``lo`` contributes its
    lower edge ``lo − _HALF`` and the highest integer ``hi`` its upper edge ``hi + _HALF``, so
    the whole label spans ``[lo − _HALF, hi + _HALF)``. Summing per-integer
    ``[k − _HALF, k + _HALF)`` intervals over ``lo..hi`` collapses to exactly this span
    because the integers are contiguous, which is what makes a full ladder tile the line
    without gaps or overlaps.

    ``open_lo`` / ``open_hi`` mark open-ended tail buckets (``≤X`` / ``≥Y``): the open end uses
    the ∓∞ sentinel and only the closed end carries a ``±_HALF`` offset.

    The edge offset lives in exactly one place (``_HALF``); see the module docstring on the
    04-06 human-verify lock.
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
    """Probability mass in one continuous bucket by CDF differencing (D-04, Pattern 2).

    Returns ``Φ_blend(hi) − Φ_blend(lo)`` with ``Φ_blend`` the blended-Gaussian CDF, computed
    via the erf-based ``normal_cdf`` promoted public in :mod:`weatherquant.calibrate.crps`
    (never a second erf — Pitfall 6). ``lo``/``hi`` are the CONTINUOUS edges already offset by
    ``±_HALF`` (see :func:`integers_in_bucket`).

    ``open_hi=True`` collapses the upper edge to the ``1.0`` tail (mass up to ``+∞``);
    ``open_lo=True`` collapses the lower edge to the ``0.0`` tail (mass from ``−∞``).

    Fails loud (ASVS V5 / threat T-04-09, mirroring commit ``93202d8``): ``sigma`` must be
    strictly positive and finite, and ``mu`` finite — a non-finite or non-positive input
    raises rather than silently returning a NaN probability.
    """
    if not math.isfinite(mu):
        raise ValueError(f"bucket_prob: mu must be finite, got {mu!r}.")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"bucket_prob: sigma must be finite and > 0, got {sigma!r}.")

    upper = 1.0 if open_hi else _normal_cdf_scalar((hi - mu) / sigma)
    lower = 0.0 if open_lo else _normal_cdf_scalar((lo - mu) / sigma)
    return upper - lower


def bucket_probs(
    mu: float,
    sigma: float,
    ladder: Sequence[tuple[float, float, bool, bool]],
) -> NDArray[np.float64]:
    """Probabilities across a full bucket ladder, summing to ~1 (D-04, Pattern 2).

    ``ladder`` is a sequence of ``(lo, hi, open_lo, open_hi)`` continuous buckets tiling the
    line (including the open ``≤X`` / ``≥Y`` tails). Returns one probability per bucket as a
    float array. When the ladder tiles ``(−∞, ∞)`` with no gaps or overlaps — the property
    :func:`integers_in_bucket` guarantees — the array sums to ~1 (asserted by
    ``tests/test_buckets.py -k sum`` within 1e-9). ``mu``/``sigma`` are guarded by
    :func:`bucket_prob` per bucket (fail loud on non-finite / σ≤0).
    """
    return np.array(
        [bucket_prob(mu, sigma, lo, hi, open_lo, open_hi) for lo, hi, open_lo, open_hi in ladder],
        dtype=np.float64,
    )


def parse_ticker(ticker: str) -> tuple[int, int]:
    """Pure ``ticker → (lo, hi)`` integer-degree edge parser, fail-loud (D-06 — Wave 1).

    Parses a Kalshi ``KXHIGH{CITY}`` range label/ticker into its inclusive integer-degree
    bounds. Malformed input raises (never silently defaults an edge — ASVS V5); structured
    ``floor_strike``/``cap_strike`` are preferred over label parsing where available.
    """
    raise NotImplementedError("parse_ticker is implemented in Wave 1 (04-03).")
