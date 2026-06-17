"""The shared params->Gaussian link (D-14) and the CRPS chain-rule to (a,b,c,d) (D-02).

This module is the SINGLE source of truth for turning the 4 EMOS/NGR parameters into a
predictive Gaussian. :func:`predict` is the link Phase 4 reuses **verbatim** when it blends
and prices (D-14) — so the calibration fit and the downstream pricing apply the exact same
``params -> (mu, sigma)`` mapping, with no risk of a divergent re-implementation.

The NGR parameterization (D-02) is::

    mu      = a + b * m                              (m = ensemble mean forecast, °F)
    var_raw = c^2 + d^2 * s2                          (s2 = ensemble variance)
    var     = max(sigma_floor^2, var_raw)            (the σ-floor clamp, D-09)
    sigma   = sqrt(var)

Squaring ``c`` and ``d`` makes the variance non-negative *by construction* — no constrained
optimization, no projection (D-02). The additive ``sigma_floor`` blocks the degenerate
over-confidence that would otherwise blow up Phase-4 Kelly sizing (D-09). For a deterministic
single-member model ``s2 == 0`` ⇒ ``var = c^2`` a fitted constant and ``d`` is inactive
(its gradient is identically 0) — **expected, not a bug** (D-02 / RESEARCH Pitfall 2).

:func:`param_grads` chain-rules the closed-form CRPS gradient ``(d/dmu, d/dsigma)`` from
:func:`weatherquant.calibrate.crps.crps_norm_grad` onto ``(a, b, c, d)`` for the per-stratum
mean-CRPS objective. The subtle part is the clamp: ``var = max(...)`` is non-differentiable
at the kink, so when the floor is ACTIVE (``var_raw <= sigma_floor^2``) ``sigma`` is constant
and ``dsigma/dc = dsigma/dd = 0`` — the variance-param gradient must be masked to 0
(RESEARCH Pitfall 1), or the optimizer takes a phantom step it can never escape.

Pure NumPy + ``math`` only (via crps.py) — no scipy/sklearn (the AST guard enforces it).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from weatherquant.calibrate.crps import crps_norm_grad

__all__ = ["predict", "param_grads"]

# A params bundle is (a, b, c, d, sigma_floor): the 4 fitted EMOS params plus the σ-floor.
Params = tuple[float, float, float, float, float]


def predict(
    params: Params,
    mean_f: NDArray[np.float64],
    var_f: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reconstruct the predictive Gaussian ``(mu, sigma)`` from params (D-14, D-02).

    The single shared params->Gaussian link, reused verbatim by Phase 4's blending/pricing
    so calibration and pricing never diverge. ``params`` is ``(a, b, c, d, sigma_floor)``;
    ``mean_f`` is the ensemble-mean forecast (°F) and ``var_f`` the ensemble variance
    (``0`` for a deterministic model). Applies ``mu = a + b*mean_f`` and
    ``sigma = sqrt(max(sigma_floor^2, c^2 + d^2*var_f))`` (the σ-floor clamp, D-09).
    """
    a, b, c, d, sigma_floor = params
    mu = a + b * mean_f
    var_raw = c * c + d * d * var_f
    var = np.maximum(sigma_floor * sigma_floor, var_raw)
    sigma = np.sqrt(var)
    return mu, sigma


def param_grads(
    a: float,
    b: float,
    c: float,
    d: float,
    sigma_floor: float,
    m: NDArray[np.float64],
    s2: NDArray[np.float64],
    y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Mean-CRPS gradient w.r.t. ``(a, b, c, d)`` over a stratum (D-02 chain-rule).

    Chain-rules ``(d CRPS/d mu, d CRPS/d sigma)`` (from
    :func:`weatherquant.calibrate.crps.crps_norm_grad`) onto the 4 EMOS params under the
    D-02 link, returning ``np.array([g_a, g_b, g_c, g_d])`` averaged over the stratum's
    samples (the mean-CRPS objective the optimizer minimizes).

    Mean part: ``d mu/d a = 1`` and ``d mu/d b = m``. Variance part (floor INACTIVE):
    ``d sigma/d c = c/sigma`` and ``d sigma/d d = d*s2/sigma``. When the σ-floor is ACTIVE
    (``c^2 + d^2*s2 <= sigma_floor^2``) ``sigma`` is constant, so ``d sigma/d c`` and
    ``d sigma/d d`` are masked to 0 (RESEARCH Pitfall 1). For ``s2 == 0`` (deterministic
    model) ``g_d`` is identically 0 — expected (D-02). Verified end-to-end against finite
    differences by ``tests/test_crps_gradient.py``.
    """
    mu = a + b * m
    var_raw = c * c + d * d * s2
    var = np.maximum(sigma_floor * sigma_floor, var_raw)
    sigma = np.sqrt(var)

    d_mu, d_sigma = crps_norm_grad(mu, sigma, y)

    floor_active = var_raw <= sigma_floor * sigma_floor
    dsig_dc = np.where(floor_active, 0.0, c / sigma)
    dsig_dd = np.where(floor_active, 0.0, d * s2 / sigma)

    g_a = d_mu  # d mu / d a = 1
    g_b = d_mu * m  # d mu / d b = m
    g_c = d_sigma * dsig_dc
    g_d = d_sigma * dsig_dd  # 0 for deterministic models (s2 == 0) — D-02, expected

    n = len(y)
    return np.array(
        [g_a.sum(), g_b.sum(), g_c.sum(), g_d.sum()], dtype=float
    ) / n
