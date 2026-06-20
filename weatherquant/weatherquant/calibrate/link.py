"""The shared params->Gaussian link (D-14) and the CRPS chain-rule to (a,b,c,d) (D-02).

Single source of truth for the 4 EMOS/NGR params → predictive Gaussian. :func:`predict` is
reused VERBATIM by Phase-4 blending/pricing so the fit and pricing never diverge (D-14). The
NGR parameterization (D-02): ``mu = a + b*m``; ``var = max(sigma_floor^2, c^2 + d^2*s2)``.
Squaring c, d makes variance non-negative by construction; ``s2 == 0`` (deterministic) leaves
``d`` inactive (gradient ≡ 0) — expected, not a bug (D-02).

LOAD-BEARING (Pitfall 1 / D-02): :func:`param_grads` chain-rules the CRPS gradient onto
``(a,b,c,d)``, but ``var = max(...)`` has a kink — when the floor is ACTIVE
(``var_raw <= sigma_floor^2``) σ is constant so ``dsigma/dc = dsigma/dd = 0`` MUST be masked,
else the optimizer takes a phantom step it can never escape.

Pure NumPy + ``math`` only — no scipy/sklearn (the AST guard enforces it).
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

    The shared link reused verbatim by Phase-4 pricing. ``params`` is ``(a, b, c, d, sigma_floor)``;
    applies ``mu = a + b*mean_f`` and ``sigma = sqrt(max(sigma_floor^2, c^2 + d^2*var_f))`` (D-09).
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

    Chain-rules the CRPS gradient onto the 4 EMOS params, returning ``[g_a, g_b, g_c, g_d]``
    averaged over the stratum. When the σ-floor is ACTIVE (``c^2 + d^2*s2 <= sigma_floor^2``) σ
    is constant, so ``dsigma/dc`` and ``dsigma/dd`` are masked to 0 (Pitfall 1; for ``s2 == 0``
    ``g_d`` is identically 0 — expected). Verified by ``tests/test_crps_gradient.py``.
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
