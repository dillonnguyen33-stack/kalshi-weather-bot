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

import numpy as np
from numpy.typing import NDArray

__all__ = ["integers_in_bucket", "bucket_prob", "bucket_probs", "parse_ticker"]

# The single half-degree bucket-edge offset (D-05 / RESEARCH Pitfall 1): integer degree ``k``
# owns the continuous interval ``[k − _HALF, k + _HALF)``. Centralized here so a one-place
# change (after the live-market human-verify checkpoint) re-maps every bucket consistently.
_HALF = 0.5


def integers_in_bucket(
    lo: int,
    hi: int,
    open_lo: bool = False,
    open_hi: bool = False,
) -> NDArray[np.float64]:
    """Continuous ``[k − _HALF, k + _HALF)`` edges for a labeled bucket (D-05 — Wave 1).

    Maps the inclusive integer degrees a bucket label covers (``lo``..``hi``) to the
    continuous half-degree interval used for CDF differencing. ``open_lo``/``open_hi`` mark
    open-ended tail buckets (``≤X`` / ``≥Y``).
    """
    raise NotImplementedError("integers_in_bucket is implemented in Wave 1 (04-03).")


def bucket_prob(
    mu: float,
    sigma: float,
    lo: float,
    hi: float,
    open_lo: bool = False,
    open_hi: bool = False,
) -> float:
    """Probability mass in one continuous bucket by CDF differencing (D-04 — Wave 1).

    ``Φ_blend(hi) − Φ_blend(lo)`` using the erf-based ``normal_cdf`` from
    :mod:`weatherquant.calibrate.crps`. ``open_lo``/``open_hi`` collapse the respective edge
    to the 0/1 tail for open-ended buckets.
    """
    raise NotImplementedError("bucket_prob is implemented in Wave 1 (04-03).")


def bucket_probs(
    mu: float,
    sigma: float,
    ladder: object,
) -> NDArray[np.float64]:
    """Probabilities across a full bucket ladder, summing to ~1 (D-04 — Wave 1).

    ``ladder`` is a sequence of ``(lo, hi, open_lo, open_hi)`` buckets tiling the line
    (incl. open tails); returns one probability per bucket. The ladder is asserted to sum to
    ~1 by ``tests/test_buckets.py -k sum``.
    """
    raise NotImplementedError("bucket_probs is implemented in Wave 1 (04-03).")


def parse_ticker(ticker: str) -> tuple[int, int]:
    """Pure ``ticker → (lo, hi)`` integer-degree edge parser, fail-loud (D-06 — Wave 1).

    Parses a Kalshi ``KXHIGH{CITY}`` range label/ticker into its inclusive integer-degree
    bounds. Malformed input raises (never silently defaults an edge — ASVS V5); structured
    ``floor_strike``/``cap_strike`` are preferred over label parsing where available.
    """
    raise NotImplementedError("parse_ticker is implemented in Wave 1 (04-03).")
