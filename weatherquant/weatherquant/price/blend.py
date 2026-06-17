"""Accuracy-weighted Gaussian Vincentization — the blend (PRC-01, D-01/D-02/D-03).

Quantile-average the Phase-3 calibrated per-model Gaussians into ONE predictive Gaussian.
For Gaussians, quantile averaging has the exact closed form
``N(μ_blend = Σwᵢμᵢ, σ_blend = Σwᵢσᵢ)`` — a *weighted mean of the std-devs*, NOT
``sqrt(Σwᵢσᵢ²)`` and NOT the (overdispersed) linear-mixture variance (D-01, RESEARCH
Pitfall 2). This makes the σ-monotonicity invariant ``σ_blend ≤ max(σᵢ)`` true **by
construction**, which ``tests/test_blend.py -k monoton`` guards.

Per-model ``(μᵢ, σᵢ)`` come from :func:`weatherquant.calibrate.link.predict` reused verbatim
(D-14/D-15) — this module never re-derives the params→Gaussian link. Accuracy weights derive
from the per-stratum OOS CRPS Phase 3 persists (``crps_oos``): lower CRPS ⇒ higher weight,
via normalized inverse-CRPS with a floor so no model fully dominates or drops (D-02). A
missing model drops out and the survivors renormalize; a NULL-``crps_oos`` (pure-pooled fit)
falls back to its pooled-parent CRPS or equal weight (D-03, RESEARCH Pitfall 4). ``crps_oos``
is used only as a RELATIVE cross-model signal, never an absolute quality measure (Pitfall 5).

Pure NumPy + stdlib ``math`` only — no scipy/sklearn (the AST guard enforces it).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["accuracy_weights", "blend_gaussians"]

# --- Named operational constants (RESEARCH §Operational Defaults — never a magic number
# inline; every LOW/MEDIUM knob is a module-level constant Phase 6 can audit/revisit). ---

#: Lower bound on ``crps_oos`` in the inverse so a (near-)zero CRPS can't blow the weight up
#: to ±inf. RESEARCH §Operational Defaults: ``w_i ∝ 1/max(crps_oos_i, ε)``. Small enough not
#: to perturb realistic CRPS magnitudes (CRPS is in °F, ~O(1)).
CRPS_EPS: float = 1e-6

#: Per-model weight floor applied BEFORE the final renormalize so no single model fully
#: dominates or is fully dropped (RESEARCH §Operational Defaults — equal weights are hard to
#: beat, so tilt gently; a floor keeps the ensemble diverse). For ``k`` present models the
#: floor must satisfy ``k * W_MIN <= 1``; with the in-scope model set (≤ ~8) ``0.05`` is safe.
W_MIN: float = 0.05


def accuracy_weights(
    crps_oos: NDArray[np.float64],
    *,
    parent_crps: NDArray[np.float64] | None = None,
    eps: float = CRPS_EPS,
    w_min: float = W_MIN,
) -> NDArray[np.float64]:
    """Normalized inverse-CRPS weights with a floor (D-02/D-03, RESEARCH §Operational Defaults).

    Turns the per-model OOS CRPS of the models PRESENT for a given ``(city, lead, month)``
    into a weight vector that sums to 1, with lower ``crps_oos`` ⇒ higher weight via
    normalized inverse-CRPS ``w_i ∝ 1/max(crps_oos_i, eps)``. A per-model floor ``w_min`` is
    applied before the final renormalize so no model fully dominates or fully drops
    (RESEARCH §Operational Defaults; equal weights are hard to beat → tilt gently). Because
    only the models present are passed in, the renormalize that closes this function is also
    the dropped-model renormalization (D-03 / RESEARCH Pitfall 4): a missing model is simply
    absent from ``crps_oos`` and the survivors' weights renormalize to sum to 1.

    A NULL/NaN ``crps_oos`` entry (a pure-pooled fit has no own OOS score) falls back to the
    supplied pooled-parent CRPS (``parent_crps[i]``) if finite, else to that model's
    equal-weight share — never NaN, never a silent zero (D-03, threat T-04-04). If EVERY
    present model is NULL (and no usable parent) the result is exactly equal weights.

    ``crps_oos`` is used ONLY as a RELATIVE cross-model accuracy signal, never as an absolute
    quality measure of the persisted (possibly season-pooled) params — its provenance is an
    unpooled train-slice fit (RESEARCH Pitfall 5; ``evaluate.OOSResult`` docstring).

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

    # --- Resolve NULL/NaN entries (D-03). Substitute a finite pooled-parent CRPS where
    # available; the still-NULL set is handled as equal-weight below. ---
    null = ~np.isfinite(crps)
    if parent_crps is not None:
        parent = np.asarray(parent_crps, dtype=np.float64).ravel()
        if parent.shape[0] != n:
            raise ValueError("parent_crps must match crps_oos in length.")
        substitute = null & np.isfinite(parent)
        crps = np.where(substitute, parent, crps)
        null = ~np.isfinite(crps)

    # --- Normalized inverse-CRPS raw weights for the models with a usable CRPS. A NULL
    # model gets the mean usable raw weight (its equal-weight share) so it neither dominates
    # nor zeroes; if NO model is usable, every raw weight is equal → equal weights (D-03). ---
    raw = np.zeros(n, dtype=np.float64)
    usable = ~null
    if np.any(usable):
        raw[usable] = 1.0 / np.maximum(crps[usable], eps)
        raw[null] = float(np.mean(raw[usable]))
    else:
        raw[:] = 1.0
    raw = raw / raw.sum()

    # --- Per-model floor, then final renormalize (also the dropped-model renormalization,
    # D-03 / Pitfall 4). The floor guarantees a strictly positive normalizer (threat
    # T-04-06) and keeps every present model in the blend. ---
    floored = np.maximum(raw, w_min)
    return floored / floored.sum()


def blend_gaussians(
    mus: NDArray[np.float64],
    sigmas: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[float, float]:
    """Vincentization closed form ``N(Σwᵢμᵢ, (Σwᵢσᵢ)²)`` (D-01 — implemented in Wave 1).

    ``mus``/``sigmas``/``weights`` are 1-D arrays over the models present; ``weights`` are
    renormalized internally (handles dropped models, D-03). Returns ``(μ_blend, σ_blend)``
    with ``σ_blend = Σwᵢσᵢ`` (weighted MEAN of σ), so ``σ_blend ≤ max(σᵢ)`` by construction.
    """
    raise NotImplementedError("blend_gaussians is implemented in Wave 1 (04-02).")
