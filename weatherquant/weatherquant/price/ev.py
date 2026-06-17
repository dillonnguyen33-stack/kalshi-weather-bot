"""Fee-corrected expected value with market-midpoint shrink (PRC-03, D-08).

EV per Kalshi contract uses a model probability shrunk toward the market midpoint:
``p_used = (1 − α)·p_model + α·p_market_mid``, with ``p_market_mid = (best_bid + best_ask)/2``
passed in (no market I/O here — D-16). The EV on the achievable (taker) price is

    EV = p_used·((1 − price) − fee) − (1 − p_used)·price

with the EXACT integer-cent fee :func:`weatherquant.price.fee.exact_fee` (taker, D-09). This
re-expresses v3's ``t_ev = prob·(win − t_fee) − (1 − prob)·p`` (``win = 1 − p``) intent with
the exact fee; ``tests/test_ev.py`` asserts EV-parity against that v3 arithmetic on a shared
example, and ``-k shrink`` asserts ``p_used`` moves toward ``p_market_mid`` as α↑.

The market-shrink coefficient ``MARKET_SHRINK_ALPHA`` is a named module constant (RESEARCH
Operational Defaults — never a magic number inline); it is co-chosen with the Kelly λ, not
stacked blindly, and is revisited only via Phase 6's pre-registered process.

Pure NumPy + stdlib ``math`` only — no scipy/sklearn (the AST guard enforces it).
"""

from __future__ import annotations

__all__ = ["MARKET_SHRINK_ALPHA", "p_used", "bucket_ev"]

# Linear market-shrink weight toward the market midpoint (D-08, RESEARCH Operational
# Defaults, α≈0.2 — LOW confidence, co-tuned with Kelly λ via Phase 6, never stacked blindly).
MARKET_SHRINK_ALPHA = 0.2


def p_used(
    p_model: float,
    p_market_mid: float,
    alpha: float = MARKET_SHRINK_ALPHA,
) -> float:
    """Model probability shrunk toward the market midpoint (D-08 — Wave 2).

    ``(1 − alpha)·p_model + alpha·p_market_mid``. ``alpha=0`` trusts the model fully;
    ``alpha=1`` defers entirely to the market. ``tests/test_ev.py -k shrink`` asserts the
    result moves toward ``p_market_mid`` as ``alpha`` rises. Guards probabilities ∈ [0, 1].
    """
    raise NotImplementedError("p_used is implemented in Wave 2 (04-05).")


def bucket_ev(
    p_model: float,
    p_market_mid: float,
    price: float,
    alpha: float = MARKET_SHRINK_ALPHA,
) -> float:
    """Fee-corrected per-contract EV on the taker price (D-08 — Wave 2).

    ``EV = p_used·((1 − price) − fee) − (1 − p_used)·price`` with the exact taker fee
    ``exact_fee(1, price)`` and ``p_used`` shrunk toward ``p_market_mid``. Matches v3's
    ``t_ev`` intent (parity-tested) but with the exact integer-cent fee. Guards
    ``price ∈ [0, 1]``, probabilities finite (ASVS V5).
    """
    raise NotImplementedError("bucket_ev is implemented in Wave 2 (04-05).")
