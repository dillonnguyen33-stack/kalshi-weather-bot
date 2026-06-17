"""LINCHPIN: finite-difference gradient-check for the Gaussian CRPS (CAL-02 / D-05).

A silent algebra error in the CRPS gradient would poison every downstream fit, every OOS
metric, and the eventual Gate-1 proof — and it would do so invisibly, since a wrong-but-
plausible gradient still "optimizes" to *something*. This test is the structural guard
(CONTEXT D-05): it central-differences the closed-form ``crps_norm`` and asserts the
analytic ``crps_norm_grad`` matches, over thousands of random ``(mu, sigma, y)`` draws.

Two checks:

1. ``test_crps_gradient_matches_finite_difference`` — the raw ``(d/dmu, d/dsigma)`` against
   central differences of ``crps_norm`` (RESEARCH verified max|fd - analytic| ~1e-8).
2. ``test_param_grads_match_finite_difference`` — the full ``(a, b, c, d)`` chain-rule
   through the D-02 link (``param_grads``), which is what actually guards the chain-rule
   algebra of Pattern 2, **including the sigma-floor-active branch** where the variance-
   param gradient must vanish. Two regimes are exercised: floor-inactive (gradient flows to
   c, d) and floor-active (gradient to c, d is identically 0).
"""

from __future__ import annotations

import numpy as np

from weatherquant.calibrate.crps import crps_norm, crps_norm_grad
from weatherquant.calibrate.link import param_grads, predict


def test_crps_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    h = 1e-6
    for _ in range(2000):
        mu = rng.uniform(-30, 30)
        sigma = rng.uniform(0.3, 8.0)
        y = rng.uniform(-40, 40)
        d_mu, d_sigma = crps_norm_grad(mu, sigma, y)
        g_mu = (crps_norm(mu + h, sigma, y) - crps_norm(mu - h, sigma, y)) / (2 * h)
        g_si = (crps_norm(mu, sigma + h, y) - crps_norm(mu, sigma - h, y)) / (2 * h)
        assert abs(float(d_mu) - g_mu) < 1e-5
        assert abs(float(d_sigma) - g_si) < 1e-5


def _mean_crps_for_params(a, b, c, d, sigma_floor, m, s2, y):
    """Mean CRPS over a stratum as a scalar function of (a, b, c, d) — the fit objective."""
    mu, sigma = predict((a, b, c, d, sigma_floor), m, s2)
    return float(crps_norm(mu, sigma, y).mean())


def test_param_grads_match_finite_difference():
    """The (a,b,c,d) chain-rule (incl. floor-active mask) matches central differences."""
    rng = np.random.default_rng(7)
    h = 1e-6
    n = 64

    for _ in range(300):
        m = rng.uniform(40, 90, n)
        # Exercise BOTH regimes: ensemble (s2>0) and deterministic (s2=0).
        s2 = rng.uniform(0.0, 9.0, n) if rng.random() < 0.5 else np.zeros(n)
        a, b = rng.uniform(-5, 5), rng.uniform(0.5, 1.5)
        c, d = rng.uniform(0.2, 4.0), rng.uniform(0.0, 2.0)
        sigma_floor = rng.uniform(0.3, 1.5)
        y = rng.uniform(40, 90, n)

        analytic = param_grads(a, b, c, d, sigma_floor, m, s2, y)

        base = (a, b, c, d, sigma_floor, m, s2, y)
        for i, name in enumerate("abcd"):
            params = list((a, b, c, d))
            params[i] += h
            up = _mean_crps_for_params(*params, sigma_floor, m, s2, y)
            params[i] -= 2 * h
            dn = _mean_crps_for_params(*params, sigma_floor, m, s2, y)
            fd = (up - dn) / (2 * h)
            assert abs(float(analytic[i]) - fd) < 1e-5, (
                f"chain-rule mismatch on d/d{name}: analytic={analytic[i]} fd={fd}"
            )
        del base


def test_param_grads_floor_active_zeroes_variance_gradient():
    """When the sigma-floor clamp is active, the gradient w.r.t. c and d must be 0.

    Forcing tiny variance params below the floor (c^2 + d^2*s^2 <= sigma_floor^2) drives the
    clamp; the chain rule must mask dsig/dc and dsig/dd to 0 (RESEARCH Pitfall 1), so g_c
    and g_d are identically zero.
    """
    rng = np.random.default_rng(11)
    n = 32
    m = rng.uniform(40, 90, n)
    s2 = rng.uniform(0.0, 4.0, n)
    y = rng.uniform(40, 90, n)
    # c, d tiny relative to a large floor => floor active for all samples.
    g = param_grads(1.0, 1.0, 1e-3, 1e-3, sigma_floor=2.0, m=m, s2=s2, y=y)
    assert g[2] == 0.0
    assert g[3] == 0.0


def test_param_grads_deterministic_model_zero_d_gradient():
    """For a deterministic model (s2=0) the gradient w.r.t. d is identically 0 (D-02)."""
    rng = np.random.default_rng(13)
    n = 48
    m = rng.uniform(40, 90, n)
    s2 = np.zeros(n)
    y = rng.uniform(40, 90, n)
    g = param_grads(0.5, 1.0, 2.0, 0.7, sigma_floor=0.5, m=m, s2=s2, y=y)
    assert g[3] == 0.0
