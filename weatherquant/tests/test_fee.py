"""Golden exact-fee tests (PRC-03, D-07) — RED until Wave 1 (04-04) implements ``exact_fee``.

The Kalshi taker fee is ``ceil_to_next_cent(0.07·n·p·(1−p))`` over the order's ``n`` contracts,
rounded UP once per order (RESEARCH §Kalshi Fee Schedule, Pitfall 3). These golden rows pin
the exact integer-cent arithmetic with closed-form expected values (mirroring
``test_crps_value.py``'s reference-point style):

* ``exact_fee(20, 0.60)`` → ``0.07·20·0.60·0.40 = $0.336`` → ceil → **$0.34**
* ``exact_fee(100, 0.50)`` → ``0.07·100·0.50·0.50 = $1.75`` → **$1.75**
* ``exact_fee(1, 0.50)`` → ``0.07·1·0.25 = $0.0175`` → ceil → **$0.02** (single-contract floor)

These are ``xfail`` (the stub raises ``NotImplementedError``) so Wave 1 flips them GREEN
without renaming.
"""

from __future__ import annotations

import pytest

from weatherquant.price.fee import exact_fee


@pytest.mark.xfail(reason="Wave 1 (04-04) implements exact_fee", strict=False)
def test_exact_fee_golden_twenty_at_sixty():
    # 0.07 * 20 * 0.60 * 0.40 = 0.336 -> ceil to next cent -> 0.34
    assert exact_fee(20, 0.60) == 0.34


@pytest.mark.xfail(reason="Wave 1 (04-04) implements exact_fee", strict=False)
def test_exact_fee_golden_hundred_at_fifty():
    # 0.07 * 100 * 0.50 * 0.50 = 1.75 exactly
    assert exact_fee(100, 0.50) == 1.75


@pytest.mark.xfail(reason="Wave 1 (04-04) implements exact_fee", strict=False)
def test_exact_fee_single_contract_rounds_up_to_two_cents():
    # 0.07 * 1 * 0.50 * 0.50 = 0.0175 -> ceil to next cent -> 0.02 (sub-cent floor effect)
    assert exact_fee(1, 0.50) == 0.02
