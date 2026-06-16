"""CRPS closed-form value checks against known references (CAL-02 / D-04).

Complements the gradient-check linchpin: a correct gradient of a *wrong* value would still
pass the finite-difference test, so the value itself is pinned to two independent reference
points derived directly from the closed form ``CRPS(N(mu,sigma), y) =
sigma*[z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi)]``:

* ``z = 0`` (forecast mean == obs): the bracket collapses to ``2*phi(0) - 1/sqrt(pi)``, so
  ``CRPS = sigma*(2*phi(0) - 1/sqrt(pi))`` exactly. Checked to <1e-12, the floor of
  ``math.erf``-double-precision Phi.
* An infinitely sharp, correct forecast (``sigma -> 0`` with ``y == mu``): CRPS -> 0.
"""

from __future__ import annotations

import math

from weatherquant.calibrate.crps import crps_norm


def test_crps_value_z_zero_reference():
    # z=0, unit sigma: CRPS = sigma * (2*phi(0) - 1/sqrt(pi)), phi(0) = 1/sqrt(2*pi).
    expected = 1.0 * (2.0 * (1.0 / math.sqrt(2.0 * math.pi)) - 1.0 / math.sqrt(math.pi))
    assert abs(float(crps_norm(0.0, 1.0, 0.0)) - expected) < 1e-12


def test_crps_value_z_zero_scales_with_sigma():
    # The z=0 value is linear in sigma — confirm the prefactor, not just the unit case.
    base = 2.0 * (1.0 / math.sqrt(2.0 * math.pi)) - 1.0 / math.sqrt(math.pi)
    for sigma in (0.5, 2.0, 7.5):
        expected = sigma * base
        assert abs(float(crps_norm(3.0, sigma, 3.0)) - expected) < 1e-12


def test_crps_sharp_correct_forecast_near_zero():
    # Infinitely sharp and correct: sigma -> 0 with y == mu => CRPS -> 0.
    assert float(crps_norm(5.0, 1e-6, 5.0)) < 1e-5
