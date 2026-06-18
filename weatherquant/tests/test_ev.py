"""Fee-corrected EV tests (PRC-03, D-08) — GREEN as of Wave 2 (04-04, this plan).

Covers the two EV behaviors from the VALIDATION map:

* EV parity — ``bucket_ev`` matches v3's ``t_ev = prob·(win − fee) − (1−prob)·p`` intent
  (``win = 1 − p``) on a shared example, but with the EXACT integer-cent fee.
* ``-k shrink`` — ``p_used`` moves toward ``p_market_mid`` as α rises.

The stub names match the ``-k`` selectors in 04-VALIDATION.md and are flipped GREEN here
without renaming. Guard/no-I/O/no-duplicate-fee tests lock the fail-loud and one-fee-source
contract (T-04-11, D-16, ev never re-implements the fee).
"""

from __future__ import annotations

import math

import pytest

from weatherquant.price.ev import MARKET_SHRINK_ALPHA, bucket_ev, p_used


def test_ev_parity_with_v3_intent_exact_fee():
    from weatherquant.price.fee import exact_fee

    p, price = 0.62, 0.55
    fee = exact_fee(1, price)
    # v3 intent re-expressed with the exact fee: EV = p·((1−price)−fee) − (1−p)·price.
    expected = p * ((1.0 - price) - fee) - (1.0 - p) * price
    # alpha=0 ⇒ p_used == p_model, so bucket_ev reduces to the v3 arithmetic exactly.
    assert bucket_ev(p, p, price, alpha=0.0) == pytest.approx(expected)


def test_ev_shrink_moves_toward_market_mid():
    p_model, p_mid = 0.80, 0.50
    near = p_used(p_model, p_mid, alpha=0.1)
    far = p_used(p_model, p_mid, alpha=0.5)
    # As alpha rises, p_used moves from the model toward the market midpoint.
    assert abs(far - p_mid) < abs(near - p_mid)
    assert p_used(p_model, p_mid, alpha=0.0) == pytest.approx(p_model)
    assert p_used(p_model, p_mid, alpha=1.0) == pytest.approx(p_mid)


def test_p_used_default_alpha_is_the_named_constant():
    # The default shrink weight is the one named module constant (one-named-seam, D-08).
    assert MARKET_SHRINK_ALPHA == 0.2
    p_model, p_mid = 0.80, 0.50
    assert p_used(p_model, p_mid) == pytest.approx(
        (1.0 - MARKET_SHRINK_ALPHA) * p_model + MARKET_SHRINK_ALPHA * p_mid
    )


def test_p_used_is_monotone_toward_mid_across_alpha_sweep():
    p_model, p_mid = 0.80, 0.50
    prev = p_model
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        cur = p_used(p_model, p_mid, alpha=a)
        # Monotonically non-increasing from p_model down toward p_mid as alpha rises.
        assert cur <= prev + 1e-12
        assert abs(cur - p_mid) <= abs(prev - p_mid) + 1e-12
        prev = cur


def test_bucket_ev_uses_shrunk_prob_on_taker_price():
    from weatherquant.price.fee import exact_fee

    p_model, p_mid, price, alpha = 0.80, 0.50, 0.55, 0.2
    pu = p_used(p_model, p_mid, alpha)
    fee = exact_fee(1, price)
    expected = pu * ((1.0 - price) - fee) - (1.0 - pu) * price
    assert bucket_ev(p_model, p_mid, price, alpha=alpha) == pytest.approx(expected)


@pytest.mark.parametrize(
    "p_model, p_mid, price",
    [
        (1.5, 0.5, 0.5),
        (0.5, -0.1, 0.5),
        (0.5, 0.5, 1.5),
        (math.nan, 0.5, 0.5),
        (0.5, math.inf, 0.5),
        (0.5, 0.5, math.nan),
    ],
)
def test_bucket_ev_fails_loud_on_invalid_input(p_model, p_mid, price):
    with pytest.raises(ValueError):
        bucket_ev(p_model, p_mid, price)


def test_ev_module_does_not_duplicate_the_fee_coefficient():
    # The 0.07 fee coefficient must live only in fee.py; ev.py imports exact_fee, never 0.07.
    import pathlib

    import weatherquant.price.ev as ev_mod

    src = pathlib.Path(ev_mod.__file__).read_text(encoding="utf-8")
    assert "0.07" not in src
    assert "from weatherquant.price.fee import" in src
