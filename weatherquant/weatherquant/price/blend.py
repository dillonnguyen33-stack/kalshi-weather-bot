"""Accuracy-weighted Gaussian Vincentization — the blend (D-01/D-02/D-03).

LOAD-BEARING NUMERICS: σ_blend = weighted MEAN of σ (Σwᵢσᵢ), NOT sqrt(Σwᵢσᵢ²) — the linear
mix is overdispersed (D-01, Pitfall 2). This keeps σ_blend ≤ max(σᵢ) by construction.

Per-model ``(μᵢ, σᵢ)`` come from :func:`weatherquant.calibrate.link.predict` (D-14/D-15).
Weights are normalized inverse-CRPS over the present models' ``crps_oos`` (D-02/D-03; Pitfalls
4/5). Pure NumPy + stdlib ``math`` only — no scipy/sklearn (AST guard).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["accuracy_weights", "blend_gaussians"]

#: Floor on ``crps_oos`` in the inverse so a near-zero CRPS can't blow the weight to ±inf.
CRPS_EPS: float = 1e-6

#: Per-model weight floor applied BEFORE the final renormalize so no model dominates or drops.
#: Needs ``k * W_MIN <= 1``; safe for the in-scope model set (≤ ~8).
W_MIN: float = 0.05


def accuracy_weights(
    crps_oos: NDArray[np.float64],
    *,
    parent_crps: NDArray[np.float64] | None = None,
    eps: float = CRPS_EPS,
    w_min: float = W_MIN,
) -> NDArray[np.float64]:
    """Normalized inverse-CRPS weights over present models, with a pre-renormalize floor (D-02/D-03).

    ``w_i ∝ 1/max(crps_oos_i, eps)``, renormalized to sum to 1; passing only present models
    makes the renormalize double as the dropped-model renormalization (D-03; Pitfall 4). The
    floor is PRE-renormalize: an exact lower bound on the final weight only while the floored
    sum ``≤ 1`` (holds for the in-scope ≤ ~8 models). A NULL/NaN ``crps_oos`` falls back to a
    finite ``parent_crps[i]``, else equal-weight share — never NaN/zero (D-03, T-04-04).
    ``crps_oos`` is a RELATIVE cross-model signal only, never absolute quality (Pitfall 5).

    Parameters
    ----------
    crps_oos:
        1-D array of per-model OOS CRPS for the present models. NaN entries are NULL.
    parent_crps:
        Optional 1-D array (same length) of pooled-parent CRPS used to substitute a NULL
        ``crps_oos`` entry; a non-finite parent falls through to equal weight (D-03).
    eps, w_min:
        Inverse-CRPS floor and per-weight floor (default the module constants).

    Returns
    -------
    NDArray[np.float64]
        Weights over the present models, all finite, summing to 1 within ~1e-12.
    """
    crps = np.asarray(crps_oos, dtype=np.float64).ravel()
    n = crps.shape[0]
    if n == 0:
        raise ValueError("accuracy_weights requires at least one present model.")

    # Resolve NULL/NaN entries (D-03): substitute a finite pooled-parent CRPS where available.
    null = ~np.isfinite(crps)
    if parent_crps is not None:
        parent = np.asarray(parent_crps, dtype=np.float64).ravel()
        if parent.shape[0] != n:
            raise ValueError("parent_crps must match crps_oos in length.")
        substitute = null & np.isfinite(parent)
        crps = np.where(substitute, parent, crps)
        null = ~np.isfinite(crps)

    # Inverse-CRPS raw weights; a NULL model gets the mean usable raw weight, and if none is
    # usable every raw weight is equal → equal weights (D-03).
    raw = np.zeros(n, dtype=np.float64)
    usable = ~null
    if np.any(usable):
        raw[usable] = 1.0 / np.maximum(crps[usable], eps)
        raw[null] = float(np.mean(raw[usable]))
    else:
        raw[:] = 1.0
    raw = raw / raw.sum()

    # Per-model floor, then final renormalize (also the dropped-model renormalization, D-03 /
    # Pitfall 4); the floor guarantees a strictly positive normalizer (T-04-06).
    floored: NDArray[np.float64] = np.maximum(raw, w_min)
    weights: NDArray[np.float64] = floored / floored.sum()
    return weights


def blend_gaussians(
    mus: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float]:
    """Vincentization closed form ``N(Σwᵢμᵢ, (Σwᵢσᵢ)²)`` (D-01).

    Quantile averaging, NOT a linear density mixture: ``sigma_blend = Σwᵢσᵢ`` is the WEIGHTED
    MEAN of σ, so ``σ_blend ≤ max(σᵢ)`` by construction (T-04-05, ``-k monoton``). The FORBIDDEN
    ``sqrt(Σwᵢσᵢ²)`` (and the larger linear-mixture variance) overdisperses the blend and
    violates PRC-01 (Pitfall 2) — never take a sqrt of a weighted sum of squares.

    ``weights`` are renormalized internally, so dropped models leave a valid simplex (D-03,
    T-04-06). Component ``(μᵢ, σᵢ)`` come from :func:`weatherquant.calibrate.link.predict`.

    Parameters
    ----------
    mus, sigmas, weights:
        1-D arrays over the models present (``sigmas > 0``). ``weights`` need not pre-sum to 1.

    Returns
    -------
    tuple[float, float]
        ``(mu_blend, sigma_blend)`` with ``sigma_blend = Σwᵢσᵢ``.
    """
    mu_arr = np.asarray(mus, dtype=np.float64).ravel()
    sigma_arr = np.asarray(sigmas, dtype=np.float64).ravel()
    w_raw = np.asarray(weights, dtype=np.float64).ravel()
    if mu_arr.shape != sigma_arr.shape or mu_arr.shape != w_raw.shape:
        raise ValueError("mus, sigmas, weights must have matching shape.")
    if mu_arr.size == 0:
        raise ValueError("blend_gaussians requires at least one present model.")
    if not np.all(np.isfinite(mu_arr)) or not np.all(np.isfinite(sigma_arr)):
        raise ValueError("mus and sigmas must be finite.")

    total = w_raw.sum()
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("weights must be finite and sum to a positive value.")
    w = w_raw / total  # renormalize — dropped models already excluded (D-03)

    mu_blend = float(np.dot(w, mu_arr))
    sigma_blend = float(np.dot(w, sigma_arr))  # weighted MEAN of σ — NOT sqrt(Σwσ²)
    return mu_blend, sigma_blend
