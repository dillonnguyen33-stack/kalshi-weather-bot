"""Fit-recovery tests for the per-stratum NGR fitter (CAL-01, D-06).

The core validation of the calibration engine without the deferred D-15 backfill: on
synthetic Gaussian data drawn from the TRUE predictive law (the ``synthetic_stratum`` /
``synthetic_stratum_deterministic`` fixtures in ``conftest.py``), a correct
``fit_stratum`` must recover the known ``(a, b, |c|, |d|)``. The signs of ``c`` and ``d``
are free (only ``c²``, ``d²`` enter the variance link, D-02), so recovery is asserted on
absolute values.

Two regimes:

* **Ensemble** (``s2 > 0``): all four params are identifiable — ``(a, b, |c|, |d|)`` must
  land within the RESEARCH §"Synthetic fit-recovery test" tolerances.
* **Deterministic** (``s2 == 0``): ``σ² = c²`` a fitted constant, so ``(a, b, c)`` are
  recovered while ``d``'s gradient is identically 0 ⇒ ``d`` stays at its init value
  (D-02 / RESEARCH Pitfall 2 — expected, not a bug).

A third test guards the σ-floor convergence: a near-degenerate fit must not diverge — the
predictive σ stays ``>= sigma_floor`` on every Adam step.
"""

from __future__ import annotations

import numpy as np

from weatherquant.calibrate.emos import D0_INIT, fit_stratum
from weatherquant.calibrate.link import predict


def test_emos_fit_recovers_known_params(synthetic_stratum) -> None:
    """Ensemble stratum: fit recovers known ``(a, b, |c|, |d|)`` (CAL-01)."""
    s = synthetic_stratum
    a, b, c, d = fit_stratum(s.m, s.s2, s.y, sigma_floor=s.sigma_floor)

    assert abs(a - s.a) < 0.5
    assert abs(b - s.b) < 0.05
    # Signs of c, d are free — only c², d² enter the variance link (D-02).
    assert abs(abs(c) - abs(s.c)) < 0.5
    assert abs(abs(d) - abs(s.d)) < 0.5


def test_deterministic(synthetic_stratum_deterministic) -> None:
    """Deterministic stratum (s2=0): recovers (a,b,c); d stays at init (D-02, Pitfall 2)."""
    s = synthetic_stratum_deterministic
    assert np.all(s.s2 == 0.0)  # fixture invariant: this is the deterministic regime

    a, b, c, d = fit_stratum(s.m, s.s2, s.y, sigma_floor=s.sigma_floor)

    assert abs(a - s.a) < 0.5
    assert abs(b - s.b) < 0.05
    assert abs(abs(c) - abs(s.c)) < 0.5
    # d's gradient is identically 0 when s2==0, so d never moves from its init.
    # This is the intended behavior (D-02), NOT a convergence failure.
    assert d == D0_INIT


def test_sigma_floor_convergence_does_not_diverge() -> None:
    """A near-degenerate fit stays finite and σ never drops below sigma_floor."""
    rng = np.random.default_rng(7)
    n = 400
    m = rng.normal(70.0, 8.0, n)
    s2 = np.full(n, 4.0)
    # Near-degenerate: obs almost exactly equal the (a=0,b=1) forecast — pushes the
    # variance params toward the floor, where the gradient mask must hold the step.
    y = m + rng.normal(0.0, 1e-3, n)
    sigma_floor = 0.5

    a, b, c, d = fit_stratum(m, s2, y, sigma_floor=sigma_floor)

    assert np.isfinite([a, b, c, d]).all()
    _, sigma = predict((a, b, c, d, sigma_floor), m, s2)
    # The σ-floor clamp inside predict() guarantees this; assert it explicitly so a
    # regression that bypassed the floor would fail loudly.
    assert np.all(sigma >= sigma_floor - 1e-12)
