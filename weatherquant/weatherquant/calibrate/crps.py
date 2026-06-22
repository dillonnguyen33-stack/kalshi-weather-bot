"""Gaussian CRPS value and its closed-form analytic gradient (CAL-02 / D-04 / D-05).

The most safety-critical artifact in the core — a silent algebra error would invisibly corrupt
every fit and metric, so both value and gradient are EXACT closed form (never numerical
integration), guarded by the finite-difference test ``tests/test_crps_gradient.py``.

Closed form (Gneiting et al. 2005, MWR 133:1098; scoringRules App. A), ``z = (y - mu)/sigma``::

    CRPS(N(mu, sigma), y) = sigma * [ z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi) ]
    d CRPS / d mu    = 1 - 2*Phi(z)
    d CRPS / d sigma = 2*phi(z) - 1/sqrt(pi)

``Phi`` is built from stdlib ``math.erf`` (``0.5*(1 + erf(x/sqrt(2)))``, D-04) — scipy-free
(``scipy.stats.norm`` is fenced out by the AST guard). All functions are elementwise so a whole
stratum scores in one vectorized pass.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = ["crps_norm", "crps_norm_grad", "normal_cdf", "normal_pdf"]

# 1 / sqrt(pi): the closed-form CRPS additive constant (the -1/sqrt(pi) term).
_INV_SQRT_PI = 1.0 / math.sqrt(math.pi)
# 1 / sqrt(2): the Phi argument scale (Phi(x) = 0.5*(1 + erf(x/sqrt(2)))).
_INV_SQRT_2 = 1.0 / math.sqrt(2.0)
# 1 / sqrt(2*pi): the standard-normal pdf normalizer.
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

# Vectorized stdlib erf — scipy-free, full double precision (D-04). Strata are small (tens–
# hundreds of samples) so erf is not a hot path; np.vectorize is the simplest correct path.
_erf = np.vectorize(math.erf)


def _Phi(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standard-normal CDF via stdlib ``math.erf`` (D-04): ``0.5*(1 + erf(x/sqrt(2)))``."""
    return cast(NDArray[np.float64], 0.5 * (1.0 + _erf(x * _INV_SQRT_2)))


def _phi(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Standard-normal pdf: ``exp(-x^2/2) / sqrt(2*pi)``."""
    return cast(NDArray[np.float64], np.exp(-0.5 * x * x) * _INV_SQRT_2PI)


# Public aliases for Phase-4 bucket CDF differencing (D-04/D-14/D-15): same erf-based bodies
# under public names so ``weatherquant.price`` imports ONE source of truth and never re-derives erf.
normal_cdf = _Phi
normal_pdf = _phi


def crps_norm(
    mu: NDArray[np.float64], sigma: NDArray[np.float64], y: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Gaussian CRPS, elementwise (D-04).

    ``CRPS(N(mu, sigma), y) = sigma * [z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]``,
    ``z = (y - mu)/sigma``. Lower is better; units match ``y`` (°F, D-03). ``sigma`` must be
    strictly positive (guaranteed upstream by the σ-floor clamp in ``link.predict``).
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

    ``d CRPS/d mu = 1 - 2*Phi(z)`` and ``d CRPS/d sigma = 2*phi(z) - 1/sqrt(pi)``,
    ``z = (y - mu)/sigma``. Verified against central differences (``< 1e-5``) by
    ``tests/test_crps_gradient.py``; chain-ruled onto ``(a, b, c, d)`` by ``link.param_grads``.
    """
    z = (y - mu) / sigma
    d_mu = 1.0 - 2.0 * _Phi(z)
    d_sigma = 2.0 * _phi(z) - _INV_SQRT_PI
    return d_mu, d_sigma
