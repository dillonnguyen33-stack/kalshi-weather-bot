"""Calibration package — the pure-NumPy EMOS/NGR core (Phase 3).

This package fits a 4-parameter Gaussian predictive distribution per
``(city, model, lead, month)`` stratum by minimum-CRPS estimation (Gneiting et al.
2005). The whole core is closed-form and intentionally **pure NumPy + Python stdlib
``math.erf``** — *no scipy, no sklearn anywhere under this package*. That constraint is a
hard PROJECT.md / CLAUDE.md rule (deploy stays lightweight) and is mechanically enforced
by ``tests/test_no_forbidden_calibration_deps.py``, an AST guard scoped to this subpackage
path that rejects any ``scipy``/``sklearn`` import.

Plan 03-01 lands the math foundation:

* :mod:`weatherquant.calibrate.crps` — the Gaussian CRPS value and its closed-form
  analytic gradient ``(d/dmu = 1 - 2*Phi(z), d/dsigma = 2*phi(z) - 1/sqrt(pi))``, with the
  normal CDF ``Phi`` built from stdlib ``math.erf`` (D-04/D-05). The finite-difference
  gradient-check test is the linchpin guard against silent calibration-math corruption.
* :mod:`weatherquant.calibrate.link` — the shared ``predict(params, mean_f, var_f) ->
  (mu, sigma)`` params->Gaussian link (D-14, reused verbatim by Phase 4) and the
  ``(mu, sigma) -> (a, b, c, d)`` chain-rule ``param_grads`` (D-02), single source of truth.

Later 03 plans add the per-stratum fit (Adam + lstsq warm-start), sparse-strata
shrinkage/pooling, the OOS-beats-baseline harness, and append-only persistence.
"""

from __future__ import annotations
