"""Exact integer-cent Kalshi fee (D-07).

Taker fee ``ceil_to_next_cent(FEE_COEFF · n · p · (1 − p))`` — ceiled UP to the next whole
cent ONCE per order (never per-contract, never to-nearest) (D-07, Pitfall 3). Gate-1 sizes on
the TAKER fee only.

Pure NumPy + stdlib ``math`` only — no scipy/sklearn (AST guard).
"""

from __future__ import annotations

import math

__all__ = ["FEE_COEFF", "exact_fee"]

# Kalshi taker-fee coefficient (D-07). MEDIUM confidence — verify the KXHIGH series uses 0.07
# against a live market before real-money use.
FEE_COEFF = 0.07

# Decimals the cent-scaled fee is rounded to BEFORE the integer-cent ceiling, to absorb
# IEEE-754 noise (Pitfall 3); 9 digits is tighter than a cent yet erases the ~1e-13 error.
_CENT_SNAP_DIGITS = 9


def exact_fee(n: int, p: float, coeff: float = FEE_COEFF) -> float:
    """Exact integer-cent Kalshi taker fee for ``n`` contracts at price ``p`` (D-07).

    ``ceil(coeff · n · p · (1 − p) · 100) / 100`` — ceiled UP to the next whole cent ONCE over
    the order total. ``p`` in dollars (0..1), ``n ≥ 1``; guards fail loud (ASVS V5). Golden:
    ``(20, 0.60) → 0.34``, ``(100, 0.50) → 1.75``, ``(1, 0.50) → 0.02``.
    """
    if not math.isfinite(p):
        raise ValueError(f"exact_fee: price p must be finite, got {p!r}.")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"exact_fee: price p must be in [0, 1], got {p!r}.")
    if not math.isfinite(coeff):
        raise ValueError(f"exact_fee: coeff must be finite, got {coeff!r}.")
    if n < 1:
        raise ValueError(f"exact_fee: n must be >= 1, got {n!r}.")
    # Integer-cent ceiling over the WHOLE order once (never per-contract, never to-nearest),
    # D-07 / Pitfall 3. The cent-scaled value is snapped before ceiling so float noise (e.g.
    # 175.00000000000003 for the exact $1.75 row) does not bump the fee up a whole cent.
    cents = round(coeff * n * p * (1.0 - p) * 100.0, _CENT_SNAP_DIGITS)
    return math.ceil(cents) / 100.0
