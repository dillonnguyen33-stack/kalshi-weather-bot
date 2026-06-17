"""Fee-corrected EV tests (PRC-03, D-08) — RED until Wave 2 (04-05).

Covers the two EV behaviors from the VALIDATION map:

* EV parity — ``bucket_ev`` matches v3's ``t_ev = prob·(win − fee) − (1−prob)·p`` intent
  (``win = 1 − p``) on a shared example, but with the EXACT integer-cent fee.
* ``-k shrink`` — ``p_used`` moves toward ``p_market_mid`` as α rises.

All ``xfail`` (the stubs raise ``NotImplementedError``) so Wave 2 flips them GREEN without
renaming — the test names match the ``-k`` selectors in 04-VALIDATION.md.
"""

from __future__ import annotations

import pytest

from weatherquant.price.ev import bucket_ev, p_used


@pytest.mark.xfail(reason="Wave 2 (04-05) implements bucket_ev", strict=False)
def test_ev_parity_with_v3_intent_exact_fee():
    from weatherquant.price.fee import exact_fee

    p, price = 0.62, 0.55
    fee = exact_fee(1, price)
    # v3 intent re-expressed with the exact fee: EV = p·((1−price)−fee) − (1−p)·price.
    expected = p * ((1.0 - price) - fee) - (1.0 - p) * price
    # alpha=0 ⇒ p_used == p_model, so bucket_ev reduces to the v3 arithmetic exactly.
    assert bucket_ev(p, p, price, alpha=0.0) == pytest.approx(expected)


@pytest.mark.xfail(reason="Wave 2 (04-05) implements p_used", strict=False)
def test_ev_shrink_moves_toward_market_mid():
    p_model, p_mid = 0.80, 0.50
    near = p_used(p_model, p_mid, alpha=0.1)
    far = p_used(p_model, p_mid, alpha=0.5)
    # As alpha rises, p_used moves from the model toward the market midpoint.
    assert abs(far - p_mid) < abs(near - p_mid)
    assert p_used(p_model, p_mid, alpha=0.0) == pytest.approx(p_model)
    assert p_used(p_model, p_mid, alpha=1.0) == pytest.approx(p_mid)
