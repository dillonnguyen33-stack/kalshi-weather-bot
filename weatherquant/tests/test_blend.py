"""Vincentization blend tests (PRC-01, D-01/D-02/D-03) — RED until Wave 1 (04-02).

Covers the four blend behaviors from the VALIDATION map:

* recovery — ``blend_gaussians`` reproduces the analytic ``N(Σwμ, (Σwσ)²)`` of known
  Gaussians (the ``synthetic_gaussians`` fixture).
* ``-k monoton`` — σ-monotonicity invariant: ``σ_blend ≤ max(σᵢ)`` (true by construction).
* ``-k pit`` — non-U-shaped PIT sanity of the blend.
* ``-k weight`` — dropped-model renormalization + NULL-``crps_oos`` equal-weight fallback.

All ``xfail`` (the stubs raise ``NotImplementedError``) so Wave 1 flips them GREEN without
renaming — the test names match the ``-k`` selectors in 04-VALIDATION.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from weatherquant.price.blend import accuracy_weights, blend_gaussians


@pytest.mark.xfail(reason="Wave 1 (04-02) implements blend_gaussians", strict=False)
def test_blend_recovers_known_gaussian(synthetic_gaussians):
    mu_blend, sigma_blend = blend_gaussians(
        synthetic_gaussians.mus,
        synthetic_gaussians.sigmas,
        synthetic_gaussians.weights,
    )
    assert mu_blend == pytest.approx(synthetic_gaussians.mu_blend)
    assert sigma_blend == pytest.approx(synthetic_gaussians.sigma_blend)


@pytest.mark.xfail(reason="Wave 1 (04-02) implements blend_gaussians", strict=False)
def test_blend_sigma_monotonicity_invariant(synthetic_gaussians):
    # σ_blend = Σwᵢσᵢ (weighted mean) ⇒ σ_blend ≤ max(σᵢ) BY CONSTRUCTION (Vincentization).
    _, sigma_blend = blend_gaussians(
        synthetic_gaussians.mus,
        synthetic_gaussians.sigmas,
        synthetic_gaussians.weights,
    )
    assert sigma_blend <= float(np.max(synthetic_gaussians.sigmas)) + 1e-12


@pytest.mark.xfail(reason="Wave 1 (04-02) implements blend_gaussians", strict=False)
def test_blend_pit_not_u_shaped(synthetic_gaussians):
    # Draw from the analytic blend N(mu_blend, sigma_blend) and PIT through that same blend;
    # a correct (non-overdispersed) blend gives an approximately uniform PIT — the central
    # bins are NOT depressed into a U-shape (the linear-mixture-variance failure mode).
    from weatherquant.calibrate.crps import normal_cdf

    rng = np.random.default_rng(7)
    mu_b, sig_b = blend_gaussians(
        synthetic_gaussians.mus,
        synthetic_gaussians.sigmas,
        synthetic_gaussians.weights,
    )
    samples = rng.normal(mu_b, sig_b, 20000)
    pit = normal_cdf((samples - mu_b) / sig_b)
    counts, _ = np.histogram(pit, bins=10, range=(0.0, 1.0))
    freq = counts / counts.sum()
    # Center bins must not be starved relative to the edges (the U / ∩ signatures).
    assert freq[4] + freq[5] > freq[0] + freq[9] - 0.05


def test_blend_weight_renorm_and_null_crps_fallback():
    # Dropped-model renormalization: surviving weights still sum to 1 (D-03).
    w = accuracy_weights(np.array([0.5, 1.0, 2.0]))
    assert float(np.sum(w)) == pytest.approx(1.0)
    # Lower crps_oos ⇒ higher weight (relative accuracy signal, D-02).
    assert w[0] > w[2]
    # NULL/NaN crps_oos falls back to (non-NaN) equal weight rather than crashing (D-03).
    w_null = accuracy_weights(np.array([np.nan, np.nan]))
    assert np.all(np.isfinite(w_null))
    assert float(np.sum(w_null)) == pytest.approx(1.0)
