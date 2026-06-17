"""Exact integer-cent Kalshi fee (PRC-03, D-07/D-09).

The Kalshi taker fee is ``ceil_to_next_cent(FEE_COEFF · n · p · (1 − p))`` — computed on the
order total over its ``n`` contracts and rounded UP to the next whole cent **once** per order
(never per-contract, never to-nearest) (D-07, RESEARCH Pitfall 3). This is the EXACT
resolved fee, not v3's ``0.07·p·(1−p)`` continuous per-contract approximation (which has no
``n`` and no ceil) — the intent is re-expressed here, never ported. Golden rows
``exact_fee(20, 0.60) == 0.34`` and ``exact_fee(100, 0.50) == 1.75`` pin the formula in
``tests/test_fee.py``.

Maker vs taker is ambiguous across sources, so Gate-1 sizes on the TAKER fee (D-09); the
maker helper is exposed for completeness with a parameterized coefficient (default 0.25×
taker, "verify per market") and is deliberately NOT on the sizing path.

Magic constants are named at module scope with a docstring citing their sourcing decision
(RESEARCH Operational Defaults — never a magic number inline), mirroring the one-named-seam
discipline of ``weatherquant.calibrate.strata``.

Pure NumPy + stdlib ``math`` only — no scipy/sklearn (the AST guard enforces it).
"""

from __future__ import annotations

__all__ = ["FEE_COEFF", "exact_fee", "maker_fee"]

# Kalshi general/most-markets taker-fee coefficient (D-07, RESEARCH §Kalshi Fee Schedule):
# fee = ceil(FEE_COEFF · n · p · (1−p)). MEDIUM confidence — some product lines use 0.035;
# confirm the weather (KXHIGH) series uses 0.07 against a live market before real-money use.
FEE_COEFF = 0.07

# Default maker coefficient as a fraction of the taker fee (D-09, RESEARCH §Kalshi Fee
# Schedule). AMBIGUOUS across sources (25% / ~half / zero) — parameterized, default 0.25×,
# "verify per market"; Gate-1 sizes on taker only, so this is exposed but never on the path.
_MAKER_FRACTION_OF_TAKER = 0.25


def exact_fee(n: int, p: float, coeff: float = FEE_COEFF) -> float:
    """Exact integer-cent Kalshi taker fee for ``n`` contracts at price ``p`` (D-07 — Wave 1).

    ``ceil(coeff · n · p · (1 − p) · 100) / 100`` — rounded UP to the next whole cent once
    over the order total. ``p`` is the contract price in dollars (0..1), ``n ≥ 1``. Golden:
    ``(20, 0.60) → 0.34``, ``(100, 0.50) → 1.75``, ``(1, 0.50) → 0.02`` (sub-cent ceils to 1¢).
    Guards ``p ∈ [0, 1]``, finite, ``n ≥ 1`` at entry and fails loud (ASVS V5).
    """
    raise NotImplementedError("exact_fee is implemented in Wave 1 (04-04).")


def maker_fee(
    n: int,
    p: float,
    coeff: float = FEE_COEFF,
    maker_fraction: float = _MAKER_FRACTION_OF_TAKER,
) -> float:
    """Maker fee, parameterized off the taker fee (D-09 — Wave 1; NOT on the sizing path).

    Default ``maker_fraction × exact_fee(n, p, coeff)``. Exposed for completeness; the
    maker/taker relationship is market-dependent ("verify per market") and Gate-1 sizes on
    the taker fee only (D-09). Maker-first submission is EXE-02 / Gate-2 / out of scope.
    """
    raise NotImplementedError("maker_fee is implemented in Wave 1 (04-04).")
