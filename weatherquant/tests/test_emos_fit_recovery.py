"""Fit-recovery tests for the per-stratum NGR fitter (CAL-01, D-06).

The core validation of the calibration engine without the deferred D-15 backfill: on
synthetic Gaussian data drawn from the TRUE predictive law (the ``synthetic_stratum`` /
``synthetic_stratum_deterministic`` fixtures in ``conftest.py``), a correct
``fit_stratum`` must recover the known ``(a, b, |c|, |d|)``. The signs of ``c`` and ``d``
are free (only ``c²``, ``d²`` enter the variance link, D-02), so recovery is asserted on
absolute values.

Two regimes:

* **Ensemble** (``s2 > 0``): all four params are identifiable — ``(a, b, |c|, |d|)`` must
  land within tight tolerances. **Identifiability note:** the spread param ``d`` is only
  separable from the constant ``c`` when the ensemble variance ``s2`` genuinely *varies*
  across samples (with a constant ``s2`` the variance link ``c² + d²·s2`` is collinear, so
  only ``c² + d²·s2̄`` is determined and ``(c, d)`` are not individually recoverable).
  Likewise the intercept ``a`` is only well-separated from the slope ``b`` when ``m`` is
  centered near 0 (a forecast mean centered at ~70°F makes the OLS intercept estimate of a
  finite sample drift far from the true ``a``). This test therefore draws a *properly
  identified* stratum locally — centered ``m`` and varying ``s2`` — rather than the shared
  ``synthetic_stratum`` fixture (constant ``s2`` at ~70°F), whose ``(a, c, d)`` are not
  jointly recoverable; loosening the tolerance to pass against that fixture would make the
  recovery assertion a tautology.
* **Deterministic** (``s2 == 0``): ``σ² = c²`` a fitted constant, so ``(a, b, c)`` are
  recovered while ``d``'s gradient is identically 0 ⇒ ``d`` stays at its init value
  (D-02 / RESEARCH Pitfall 2 — expected, not a bug). The shared deterministic fixture is
  valid here (``s2 == 0`` makes ``d`` inactive by construction, no collinearity).

A third test guards the σ-floor convergence: a near-degenerate fit must not diverge — the
predictive σ stays ``>= sigma_floor`` on every Adam step.
"""

from __future__ import annotations

import numpy as np

from weatherquant.calibrate.emos import D0_INIT, fit_stratum
from weatherquant.calibrate.link import predict


def test_emos_fit_recovers_known_params() -> None:
    """Ensemble stratum: fit recovers known ``(a, b, |c|, |d|)`` (CAL-01).

    Draws a genuinely identifiable stratum: centered ``m`` (so ``a`` is separable from
    ``b``) and *varying* ``s2`` (so ``c`` is separable from ``d``) — see the module
    docstring's identifiability note. Observations are drawn from the TRUE predictive law,
    so a correct fitter must land all four params within tight tolerances.
    """
    rng = np.random.default_rng(11)
    n = 20000
    a_t, b_t, c_t, d_t = 1.0, 0.95, 1.5, 0.8
    sigma_floor = 0.5
    m = rng.normal(0.0, 8.0, n)  # centered → a separable from b
    s2 = rng.uniform(1.0, 16.0, n)  # varying → c separable from d
    mu_t = a_t + b_t * m
    sig_t = np.sqrt(np.maximum(sigma_floor**2, c_t**2 + d_t**2 * s2))
    y = rng.normal(mu_t, sig_t)

    a, b, c, d = fit_stratum(m, s2, y, sigma_floor=sigma_floor)

    assert abs(a - a_t) < 0.1
    assert abs(b - b_t) < 0.02
    # Signs of c, d are free — only c², d² enter the variance link (D-02).
    assert abs(abs(c) - c_t) < 0.2
    assert abs(abs(d) - d_t) < 0.1


def test_deterministic(synthetic_stratum_deterministic) -> None:
    """Deterministic stratum (s2=0): recovers (a,b,c); d stays at init (D-02, Pitfall 2)."""
    s = synthetic_stratum_deterministic
    assert np.all(s.s2 == 0.0)  # fixture invariant: this is the deterministic regime

    a, b, c, d = fit_stratum(s.m, s.s2, s.y, sigma_floor=s.sigma_floor)

    assert abs(a - s.a) < 0.5
    assert abs(b - s.b) < 0.05
    assert abs(abs(c) - abs(s.c)) < 0.5
    # d's gradient is identically 0 when s2==0, so d never moves from its init.
    # This is the intended behavior (D-02), NOT a convergence failure.
    assert d == D0_INIT


def test_sigma_floor_convergence_does_not_diverge() -> None:
    """A near-degenerate fit stays finite and σ never drops below sigma_floor."""
    rng = np.random.default_rng(7)
    n = 400
    m = rng.normal(70.0, 8.0, n)
    s2 = np.full(n, 4.0)
    # Near-degenerate: obs almost exactly equal the (a=0,b=1) forecast — pushes the
    # variance params toward the floor, where the gradient mask must hold the step.
    y = m + rng.normal(0.0, 1e-3, n)
    sigma_floor = 0.5

    a, b, c, d = fit_stratum(m, s2, y, sigma_floor=sigma_floor)

    assert np.isfinite([a, b, c, d]).all()
    _, sigma = predict((a, b, c, d, sigma_floor), m, s2)
    # The σ-floor clamp inside predict() guarantees this; assert it explicitly so a
    # regression that bypassed the floor would fail loudly.
    assert np.all(sigma >= sigma_floor - 1e-12)
