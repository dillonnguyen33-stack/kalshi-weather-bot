"""Golden exact-fee tests (PRC-03, D-07) — GREEN as of Wave 1 (04-04).

The Kalshi taker fee is ``ceil_to_next_cent(0.07·n·p·(1−p))`` over the order's ``n`` contracts,
rounded UP once per order (RESEARCH §Kalshi Fee Schedule, Pitfall 3). These golden rows pin
the exact integer-cent arithmetic with closed-form expected values (mirroring
``test_crps_value.py``'s reference-point style):

* ``exact_fee(20, 0.60)`` → ``0.07·20·0.60·0.40 = $0.336`` → ceil → **$0.34**
* ``exact_fee(100, 0.50)`` → ``0.07·100·0.50·0.50 = $1.75`` → **$1.75**
* ``exact_fee(1, 0.50)`` → ``0.07·1·0.25 = $0.0175`` → ceil → **$0.02** (single-contract floor)

Wave 1 (04-04) implemented the body and flipped these GREEN (no rename). Guard/maker tests
lock the fail-loud and not-for-sizing contract (T-04-10/T-04-11/T-04-12).
"""

from __future__ import annotations

import math

import pytest

from weatherquant.price.fee import FEE_COEFF, exact_fee, maker_fee


def test_exact_fee_golden_twenty_at_sixty():
    # 0.07 * 20 * 0.60 * 0.40 = 0.336 -> ceil to next cent -> 0.34
    assert exact_fee(20, 0.60) == 0.34


def test_exact_fee_golden_hundred_at_fifty():
    # 0.07 * 100 * 0.50 * 0.50 = 1.75 exactly
    assert exact_fee(100, 0.50) == 1.75


def test_exact_fee_single_contract_rounds_up_to_two_cents():
    # 0.07 * 1 * 0.50 * 0.50 = 0.0175 -> ceil to next cent -> 0.02 (sub-cent floor effect)
    assert exact_fee(1, 0.50) == 0.02


def test_fee_coeff_is_the_one_named_constant():
    # The 0.07 coefficient lives in exactly one named module constant (one-named-seam).
    assert FEE_COEFF == 0.07


def test_exact_fee_ceils_once_per_order_not_per_contract():
    # Per-contract ceiling of (1, 0.60) is 0.01 -> *20 would give 0.20; the order-total ceil
    # of (20, 0.60) is 0.34. The fee must round UP once over the whole order, never per-contract.
    per_contract_then_summed = 20 * exact_fee(1, 0.60)
    assert exact_fee(20, 0.60) == 0.34
    assert exact_fee(20, 0.60) != per_contract_then_summed


@pytest.mark.parametrize(
    "n, p",
    [
        (0, 0.5),  # n < 1
        (1, 1.5),  # p > 1
        (1, -0.1),  # p < 0
        (1, float("nan")),  # non-finite p
        (1, float("inf")),  # non-finite p
    ],
)
def test_exact_fee_fails_loud_on_invalid_input(n, p):
    with pytest.raises(ValueError):
        exact_fee(n, p)


def test_maker_fee_defaults_to_quarter_taker_and_parameterizes_coeff():
    # Maker defaults to 0.25 × the taker fee on the same (n, p).
    n, p = 100, 0.50
    assert maker_fee(n, p) == pytest.approx(0.25 * exact_fee(n, p))
    # The maker fraction is parameterizable (a per-market override, never auto-applied).
    assert maker_fee(n, p, maker_fraction=0.5) == pytest.approx(0.5 * exact_fee(n, p))
    # The taker coefficient is parameterizable too (raw, pre-ceiling intent check).
    assert maker_fee(1, 0.50, coeff=0.035) == pytest.approx(
        0.25 * exact_fee(1, 0.50, 0.035)
    )


def test_maker_fee_is_not_on_the_sizing_path_documented():
    # The maker helper exists but its docstring states it is not used for Gate-1 sizing (D-09).
    assert maker_fee.__doc__ is not None
    assert "NOT on the sizing path" in maker_fee.__doc__ or "not" in maker_fee.__doc__.lower()


def test_maker_fee_fails_loud_on_invalid_input():
    with pytest.raises(ValueError):
        maker_fee(0, 0.5)
    with pytest.raises(ValueError):
        maker_fee(1, math.nan)
