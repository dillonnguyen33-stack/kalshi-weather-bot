"""RED contract for the Gate-1 metric core (VER-01).

Asserts the eventual GREEN behavior of ``weatherquant.verify.metrics`` against synthetic Gaussian
fixtures. The imports are DEFERRED into each test body (mirroring ``test_cli_parity.py``'s "RED
until then" pattern) so collection is green while the Wave-2 implementation is RED (every call hits
``NotImplementedError`` until the bodies land).
"""

from __future__ import annotations

import numpy as np
import pytest


def _calibrated_binary(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """A perfectly-calibrated synthetic binary forecast: f ~ U(0,1), o ~ Bernoulli(f)."""
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.0, 1.0, n)
    o = (rng.uniform(0.0, 1.0, n) < f).astype(np.float64)
    return f, o


def test_brier_murphy_decomposition_identity():
    """reliability − resolution + uncertainty ≈ the binned Brier (Murphy 3-component identity)."""
    from weatherquant.verify import metrics

    f, o = _calibrated_binary(seed=1, n=20_000)
    parts = metrics.brier_murphy(f, o, n_bins=10)
    binned_brier = parts["reliability"] - parts["resolution"] + parts["uncertainty"]
    # The decomposition is of the BINNED Brier (bin-mean form), within a small binning tolerance.
    assert binned_brier == pytest.approx(metrics.brier(f, o), abs=2e-3)
    # Uncertainty is the base-rate variance o*(1-o); a ~U(0,1) forecast base rate sits near 0.5.
    assert parts["uncertainty"] == pytest.approx(0.25, abs=0.02)


def test_crps_blend_matches_calibrate_crps_norm_for_a_single_gaussian():
    """crps_blend reduces to the mean of calibrate.crps.crps_norm for one Gaussian (D-04 reuse)."""
    from weatherquant.calibrate.crps import crps_norm
    from weatherquant.verify import metrics

    rng = np.random.default_rng(7)
    mu = rng.normal(70.0, 5.0, 500)
    sigma = np.full(500, 2.5)
    y = rng.normal(mu, sigma)
    expected = float(crps_norm(mu, sigma, y).mean())
    assert metrics.crps_blend(mu, sigma, y) == pytest.approx(expected, rel=1e-9)


def test_ece_equal_count_zero_for_calibrated_positive_for_biased():
    """Equal-count ECE is ~0 for a calibrated forecast and clearly > 0 for a biased one."""
    from weatherquant.verify import metrics

    f, o = _calibrated_binary(seed=2, n=20_000)
    ece_calibrated = metrics.ece_equal_count(f, o, n_bins=10)
    # A systematic +0.2 over-forecast (clipped) is demonstrably miscalibrated.
    f_biased = np.clip(f + 0.2, 0.0, 1.0)
    ece_biased = metrics.ece_equal_count(f_biased, o, n_bins=10)
    assert ece_calibrated < 0.03
    assert ece_biased > ece_calibrated + 0.05


def test_pit_values_are_uniform_for_a_calibrated_gaussian():
    """PIT values of a correctly-specified Gaussian forecast are ~Uniform(0,1)."""
    from weatherquant.verify import metrics

    rng = np.random.default_rng(3)
    mu = rng.normal(70.0, 5.0, 50_000)
    sigma = np.full(50_000, 3.0)
    y = rng.normal(mu, sigma)
    pit = np.asarray(metrics.pit_values(y, mu, sigma))
    assert pit.min() >= 0.0 and pit.max() <= 1.0
    # Mean ≈ 0.5 and the deciles are near-uniform for a calibrated forecast.
    assert pit.mean() == pytest.approx(0.5, abs=0.01)
    deciles = np.quantile(pit, np.linspace(0.1, 0.9, 9))
    assert np.allclose(deciles, np.linspace(0.1, 0.9, 9), atol=0.02)
