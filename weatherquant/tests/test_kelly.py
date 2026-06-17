"""Fractional-Kelly sizing tests (PRC-04/05, D-10–D-13) — RED until Wave 2 (04-06).

Covers the four sizing behaviors from the VALIDATION map:

* ``-k cap`` — no sized position ever exceeds the hard cap, for arbitrary inputs.
* ``-k shrink`` — stake decreases as σ_blend widens / n_train thins.
* ``-k zero`` — a non-positive-EV side sizes to 0.
* ``-k afd`` — the AFD flag reduces the stake but NEVER zeroes it (soft haircut, PRC-05).

All ``xfail`` (the stubs raise ``NotImplementedError``) so Wave 2 flips them GREEN without
renaming — the test names match the ``-k`` selectors in 04-VALIDATION.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from weatherquant.price.kelly import kelly_fraction, stake_fraction

# A comfortably positive-EV base case for the shrink/cap/afd tests.
_P, _PRICE, _FEE = 0.75, 0.50, 0.02
_BASE = dict(n_train=500, pool_level="own:city", afd_flag=False)


@pytest.mark.xfail(reason="Wave 2 (04-06) implements stake_fraction", strict=False)
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


@pytest.mark.xfail(reason="Wave 2 (04-06) implements stake_fraction", strict=False)
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


@pytest.mark.xfail(reason="Wave 2 (04-06) implements kelly/stake", strict=False)
def test_kelly_non_positive_ev_is_zero():
    # A side whose fee-aware edge is non-positive sizes to exactly 0.
    assert kelly_fraction(0.2, 0.80, 0.02) == 0.0
    stake = stake_fraction(
        0.2, 0.80, 0.02, sigma_blend=3.0, **_BASE
    )
    assert stake == 0.0


@pytest.mark.xfail(reason="Wave 2 (04-06) implements stake_fraction", strict=False)
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
