"""Fractional-Kelly sizing tests (PRC-04/05, D-10–D-13) — GREEN as of 04-05.

Covers the four sizing behaviors from the VALIDATION map:

* ``-k cap`` — no sized position ever exceeds the hard cap, for arbitrary inputs.
* ``-k shrink`` — stake decreases as σ_blend widens / n_train thins.
* ``-k zero`` — a non-positive-EV side sizes to 0.
* ``-k afd`` — the AFD flag reduces the stake but NEVER zeroes it (soft haircut, PRC-05).

These were RED ``xfail`` stubs in Wave 0; 04-05 implements ``stake_fraction`` and flips them
GREEN without renaming — the test names still match the ``-k`` selectors in 04-VALIDATION.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from weatherquant.price.fee import exact_fee
from weatherquant.price.kelly import kelly_fraction, stake_fraction, sufficiency_ramp

# A positive-EV base case for the shrink/cap/afd tests. The edge is kept MILD on purpose
# (p just above price): a strong edge would push λ·f_kelly above the cap so EVERY shrink
# factor clips to the cap and the σ/AFD monotonicity these tests assert becomes invisible
# (both wide and narrow saturate at cap). With p=0.55, price=0.50, λ·f_kelly≈0.020 < the
# 0.025 cap, so the whole sub-cap shrink region is exercised and monotonicity is observable.
_P, _PRICE, _FEE = 0.55, 0.50, 0.02
_BASE = dict(n_train=500, pool_level="own:city", afd_flag=False)


def test_kelly_no_position_exceeds_cap():
    cap = 0.025
    rng = np.random.default_rng(11)
    for _ in range(200):
        p = float(rng.uniform(0.0, 1.0))
        price = float(rng.uniform(0.01, 0.99))
        sigma = float(rng.uniform(0.1, 20.0))
        n_train = int(rng.integers(1, 2000))
        afd = bool(rng.integers(0, 2))
        stake = stake_fraction(
            p, price, _FEE, sigma, n_train, "own:city", afd, cap=cap
        )
        assert 0.0 <= stake <= cap + 1e-12


def test_kelly_shrink_with_sigma_and_thin_data():
    wide = stake_fraction(_P, _PRICE, _FEE, sigma_blend=12.0, **_BASE)
    narrow = stake_fraction(_P, _PRICE, _FEE, sigma_blend=1.0, **_BASE)
    assert wide < narrow  # wider σ ⇒ smaller bet (D-11)

    thin = stake_fraction(
        _P, _PRICE, _FEE, sigma_blend=3.0,
        n_train=5, pool_level="own:city", afd_flag=False,
    )
    thick = stake_fraction(
        _P, _PRICE, _FEE, sigma_blend=3.0,
        n_train=1000, pool_level="own:city", afd_flag=False,
    )
    assert thin < thick  # thinner n_train ⇒ smaller bet (D-11)


def test_kelly_non_positive_ev_is_zero():
    # A side whose fee-aware edge is non-positive sizes to exactly 0.
    assert kelly_fraction(0.2, 0.80, 0.02) == 0.0
    stake = stake_fraction(
        0.2, 0.80, 0.02, sigma_blend=3.0, **_BASE
    )
    assert stake == 0.0


def test_kelly_afd_reduces_but_never_zeroes():
    with_afd = stake_fraction(
        _P, _PRICE, _FEE, sigma_blend=3.0,
        n_train=500, pool_level="own:city", afd_flag=True,
    )
    without_afd = stake_fraction(
        _P, _PRICE, _FEE, sigma_blend=3.0,
        n_train=500, pool_level="own:city", afd_flag=False,
    )
    assert with_afd < without_afd  # soft haircut reduces the stake (D-12)
    assert with_afd > 0.0  # but NEVER zeroes a positive-EV bet (PRC-05)


# --- Fail-loud input guards (WR-03): match the rest of price/'s discipline ---


@pytest.mark.parametrize("bad_p", [-0.01, 1.01, float("inf"), float("nan")])
def test_kelly_fraction_rejects_out_of_range_prob(bad_p):
    with pytest.raises(ValueError):
        kelly_fraction(bad_p, 0.5, 0.02)


@pytest.mark.parametrize("bad_price", [-0.01, 1.01, float("nan")])
def test_kelly_fraction_rejects_out_of_range_price(bad_price):
    with pytest.raises(ValueError):
        kelly_fraction(0.5, bad_price, 0.02)


@pytest.mark.parametrize("bad_sigma", [0.0, -5.0, -4.9, float("nan"), float("inf")])
def test_stake_fraction_rejects_nonpositive_sigma(bad_sigma):
    # sigma_blend ≤ 0 would divide by zero at −sigma0 or amplify the pre-cap stake; fail loud.
    with pytest.raises(ValueError):
        stake_fraction(_P, _PRICE, _FEE, sigma_blend=bad_sigma, **_BASE)


def test_kelly_fraction_resolves_fee_when_omitted():
    """Omitting `fee` resolves it via exact_fee(1, price) — the one fee seam (IN-A5, D-09)."""
    p, price = 0.60, 0.50
    assert kelly_fraction(p, price) == kelly_fraction(p, price, exact_fee(1, price))


def test_sufficiency_ramp_parent_pool_haircut():
    """A parent-pooled fit gets the extra ×0.7 haircut vs an own-stratum fit (IN-A6, D-11)."""
    n = 10  # below N_REF so the ramp is sub-1 and the 0.7 factor is observable
    assert sufficiency_ramp(n, "parent:region") == 0.7 * sufficiency_ramp(n, "own:city")
