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


def accuracy_weights(crps_oos: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalized inverse-CRPS weights with a floor (D-02 — implemented in Wave 1).

    Lower ``crps_oos`` ⇒ higher weight; a floor keeps every present model in the blend so
    no single model collapses the ensemble (D-02). NULL/NaN entries fall back to equal
    weight (D-03). Returns weights that sum to 1 over the models present.
    """
    raise NotImplementedError("accuracy_weights is implemented in Wave 1 (04-02).")


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
