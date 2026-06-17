"""Gaussian CRPS value and its closed-form analytic gradient (CAL-02 / D-04 / D-05).

The single most safety-critical artifact in the calibration core. A silent algebra error
here would corrupt every fit, every out-of-sample metric, and the eventual Gate-1 proof —
invisibly, since a wrong-but-plausible CRPS still "optimizes" to something. The structural
guard is the finite-difference gradient-check test (``tests/test_crps_gradient.py``, D-05),
which is why both the value and the gradient are implemented in *exact closed form*,
never by numerical integration.

Closed form (Gneiting et al. 2005, MWR 133:1098; scoringRules App. A), with
``z = (y - mu) / sigma``::

    CRPS(N(mu, sigma), y) = sigma * [ z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]

and the exact gradient w.r.t. the predictive parameters (D-05)::

    d CRPS / d mu    = 1 - 2*Phi(z)
    d CRPS / d sigma = 2*phi(z) - 1/sqrt(pi)

The normal CDF ``Phi`` is built from stdlib ``math.erf`` — ``Phi(x) = 0.5*(1 + erf(x/sqrt(2)))``
(D-04). ``math.erf`` is correctly-rounded double precision (~1e-16), which keeps the
gradient-check tolerance entirely down at the central-difference truncation floor; a
hand-rolled polynomial erf (e.g. A&S 7.1.26, ~1.4e-7) would needlessly eat that budget.
Crucially this is **scipy-free**: ``scipy.stats.norm`` is the forbidden "natural" reach
(PROJECT.md / CLAUDE.md), and the AST guard ``tests/test_no_forbidden_calibration_deps.py``
fences it out.

All functions are elementwise over NumPy arrays (or accept Python scalars) so a whole
stratum's residuals are scored in one vectorized pass.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = ["crps_norm", "crps_norm_grad"]

# 1 / sqrt(pi): the closed-form CRPS additive constant (the -1/sqrt(pi) term).
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
# 1 / sqrt(2): the Phi argument scale (Phi(x) = 0.5*(1 + erf(x/sqrt(2)))).
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
# 1 / sqrt(2*pi): the standard-normal pdf normalizer.
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

# Vectorized stdlib erf — scipy-free, full double precision (D-04). Per-stratum sample
# counts are small (tens–hundreds), so erf is not a hot path; np.vectorize is the simplest
# correct path and keeps the gradient-check tolerance tight.
_erf = np.vectorize(math.erf)


def _Phi(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standard-normal CDF via stdlib ``math.erf`` (D-04): ``0.5*(1 + erf(x/sqrt(2)))``."""
    return cast(NDArray[np.float64], 0.5 * (1.0 + _erf(x * _INV_SQRT_2)))


def _phi(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standard-normal pdf: ``exp(-x^2/2) / sqrt(2*pi)``."""
    return cast(NDArray[np.float64], np.exp(-0.5 * x * x) * _INV_SQRT_2PI)


def crps_norm(
    mu: NDArray[np.float64], sigma: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Gaussian CRPS, elementwise (D-04).

    ``CRPS(N(mu, sigma), y) = sigma * [z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]`` with
    ``z = (y - mu) / sigma``. Lower is better; units match ``y`` (°F on the calibration
    path, D-03). ``sigma`` must be strictly positive (the σ-floor clamp in
    :func:`weatherquant.calibrate.link.predict` guarantees this upstream).
    """
    z = (y - mu) / sigma
    return cast(
        NDArray[np.float64],
        sigma * (z * (2.0 * _Phi(z) - 1.0) + 2.0 * _phi(z) - _INV_SQRT_PI),
    )


def crps_norm_grad(
    mu: NDArray[np.float64], sigma: NDArray[np.float64], y: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Exact analytic CRPS gradient ``(d/dmu, d/dsigma)``, elementwise (D-05).

    ``d CRPS / d mu = 1 - 2*Phi(z)`` and ``d CRPS / d sigma = 2*phi(z) - 1/sqrt(pi)`` with
    ``z = (y - mu) / sigma``. Verified against central differences to the finite-difference
    truncation floor by the linchpin gradient-check test (``tests/test_crps_gradient.py`` asserts
    ``< 1e-5``, D-05). Chain-ruled onto the EMOS params ``(a, b, c, d)`` by
    :func:`weatherquant.calibrate.link.param_grads`.
    """
    z = (y - mu) / sigma
    d_mu = 1.0 - 2.0 * _Phi(z)
    d_sigma = 2.0 * _phi(z) - _INV_SQRT_PI
    return d_mu, d_sigma
