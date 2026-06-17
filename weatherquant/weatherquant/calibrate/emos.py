"""Per-stratum EMOS/NGR fit: lstsq warm-start + pure-NumPy Adam (CAL-01, D-06).

This is the calibration engine's optimizer. For one ``(city, model, lead, month)`` stratum
it fits the four NGR parameters ``(a, b, c, d)`` that minimize the **mean Gaussian CRPS** of
the predictive law under the D-02 link::

    mu      = a + b * m                              (m = ensemble mean forecast, °F)
    sigma^2 = max(sigma_floor^2, c^2 + d^2 * s2)     (s2 = ensemble variance)

The fit is deliberately small and near-convex, so no scipy/sklearn is needed (the AST guard
``tests/test_no_forbidden_calibration_deps.py`` enforces this):

1. **Warm-start (D-06).** ``(a0, b0)`` from ``np.linalg.lstsq`` regressing ``y`` on
   ``[1, m]`` — the SVD-based least-squares fit is explicitly allowed and gives Adam a
   near-optimal starting point. The variance params start from sample moments:
   ``c0 = max(residual.std(), sigma_floor)`` (so ``c0^2 >= residual variance`` and most
   strata start above the floor — RESEARCH Pitfall 1) and ``d0`` a small constant.
2. **Adam (D-06).** Hand-written Adam on the mean-CRPS objective, using
   :func:`weatherquant.calibrate.link.param_grads` for the analytic gradient (the closed-form
   CRPS gradient chain-ruled onto ``(a, b, c, d)``) and
   :func:`weatherquant.calibrate.crps.crps_norm` (via the shared :func:`predict` link) for the
   loss / convergence check. The σ-floor clamp and the variance-param gradient mask both live
   inside ``predict`` / ``param_grads`` (Pitfall 1), so each step's σ is guaranteed
   ``>= sigma_floor`` and a floor-active step never moves ``c`` / ``d`` phantomly.
3. **Convergence.** Break when ``abs(prev_loss - loss) < tol`` or after ``iters`` steps,
   returning the BEST-loss ``theta`` seen (not the last — a late Adam overshoot must never
   ship a worse-than-warm-start fit, WR-02).
4. **Fail loud (CR-02).** A non-finite loss or non-finite returned params raises
   :class:`weatherquant.ingest.errors.CalibrationError` rather than handing NaN/inf params to
   ``persist`` — a persisted NaN ``(a, b, c, d)`` would silently corrupt every downstream price.

**Deterministic models (s2 == 0).** ``d``'s gradient is identically 0 (no ensemble spread to
explain), so ``d`` stays at its init ``D0_INIT`` and ``sigma^2 = c^2`` becomes a fitted
constant — **expected, not a bug** (D-02 / RESEARCH Pitfall 2). The fit still recovers
``(a, b, c)``.

The Adam hyperparameters are named module constants / keyword args (CLI-overridable later).
They are principled, research-ranged defaults (RESEARCH §"Operational Defaults": lr≈0.05,
β₁=0.9, β₂=0.999, ε=1e-8, iters≈1000–2000, tol≈1e-7) — ASSUMPTIONS for this dataset (D-12),
not load-bearing values, and tuned only on a disjoint train/val split if ever revisited.

Pure NumPy + ``math`` (via crps.py / link.py) only — no scipy/sklearn.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from weatherquant.calibrate.crps import crps_norm
from weatherquant.calibrate.link import param_grads, predict
from weatherquant.ingest.errors import CalibrationError

__all__ = [
    "fit_stratum",
    "ADAM_LR",
    "ADAM_B1",
    "ADAM_B2",
    "ADAM_EPS",
    "ADAM_ITERS",
    "ADAM_TOL",
    "D0_INIT",
]

logger = logging.getLogger(__name__)

# --- Adam defaults (RESEARCH §"Operational Defaults"; principled ASSUMPTIONS, D-12) -------
# The fit is tiny and near-convex and lstsq warm-starts the mean params, so Adam is forgiving
# and these standard defaults converge in well under the iteration budget. CLI-overridable
# later via the keyword args; tuned (if ever) only on a train/val split disjoint from the
# Phase-6 Gate-1 slice (anti-p-hacking, D-12).
ADAM_LR: float = 0.05
ADAM_B1: float = 0.9
ADAM_B2: float = 0.999
ADAM_EPS: float = 1e-8
ADAM_ITERS: int = 2000
ADAM_TOL: float = 1e-7

# Small non-zero init for the spread param d. For ensembles it is identifiable and Adam moves
# it to the truth; for deterministic models (s2 == 0) its gradient is identically 0 so it
# stays here — exposed as a constant so the deterministic-model test can assert d is unchanged.
D0_INIT: float = 1e-3


def fit_stratum(
    m: NDArray[np.float64],
    s2: NDArray[np.float64],
    y: NDArray[np.float64],
    sigma_floor: float,
    *,
    lr: float = ADAM_LR,
    b1: float = ADAM_B1,
    b2: float = ADAM_B2,
    eps: float = ADAM_EPS,
    iters: int = ADAM_ITERS,
    tol: float = ADAM_TOL,
) -> tuple[float, float, float, float]:
    """Fit the 4 NGR params ``(a, b, c, d)`` for one stratum by minimum-CRPS (CAL-01, D-06).

    Args:
        m: per-sample ensemble-mean forecast (°F). For a deterministic single-member model
            this is just the forecast value.
        s2: per-sample ensemble variance. ``0`` (all-zeros) for a deterministic model ⇒ ``d``
            is inactive and ``sigma^2 = c^2``.
        y: verifying observations (°F), the daily-high label.
        sigma_floor: the °F σ-floor (D-09) — predictive σ is clamped to ``>= sigma_floor`` and
            the variance-param gradient is masked when the floor is active (enforced inside
            :func:`predict` / :func:`param_grads`).
        lr, b1, b2, eps, iters, tol: Adam hyperparameters (named module-constant defaults).

    Returns:
        ``(a, b, c, d)`` as Python floats — ``a, b`` the mean link ``mu = a + b*m``, ``c, d``
        the (squared) variance params ``sigma^2 = max(sigma_floor^2, c^2 + d^2*s2)``. Signs of
        ``c, d`` are free (only ``c^2``, ``d^2`` enter); for ``s2 == 0`` (deterministic) ``d``
        stays at ``D0_INIT`` (D-02 — expected, not a bug).
    """
    m = np.asarray(m, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    y = np.asarray(y, dtype=float)

    # An empty stratum has no fit — the CRPS mean is over zero samples (0/0). Fail loud rather
    # than return NaN params (IN-02): once the pooling ladder assembles strata programmatically
    # an empty group is easy to introduce, and a NaN fit must never reach persist.
    if len(y) == 0:
        raise CalibrationError("cannot fit an empty stratum (n=0)")

    # 1. lstsq warm-start for the mean params (D-06): regress y on [1, m]. SVD-based, stable,
    #    explicitly allowed (np.linalg.lstsq is not scipy/sklearn).
    design = np.column_stack([np.ones_like(m), m])
    (a0, b0), *_ = np.linalg.lstsq(design, y, rcond=None)

    # Variance params from sample moments: c0^2 >= residual variance keeps most strata above
    # the floor (Pitfall 1); d0 small (inactive for deterministic models).
    resid = y - (a0 + b0 * m)
    c0 = max(float(resid.std()), sigma_floor)
    d0 = D0_INIT

    theta = np.array([a0, b0, c0, d0], dtype=float)

    # 2. Pure-NumPy Adam on the mean-CRPS objective. Track the BEST-loss theta so a late
    #    overshoot never ships a worse fit than an earlier (or the warm-start) step (WR-02).
    mt = np.zeros(4, dtype=float)
    vt = np.zeros(4, dtype=float)
    prev_loss = np.inf
    best_loss = np.inf
    best_theta = theta.copy()

    for t in range(1, iters + 1):
        a, b, c, d = theta
        # Analytic gradient (chain-ruled, floor-masked) from the shared link.
        g = param_grads(a, b, c, d, sigma_floor, m, s2, y)

        mt = b1 * mt + (1.0 - b1) * g
        vt = b2 * vt + (1.0 - b2) * g * g
        mhat = mt / (1.0 - b1**t)
        vhat = vt / (1.0 - b2**t)
        theta = theta - lr * mhat / (np.sqrt(vhat) + eps)

        # Loss via the shared predict() link — σ-floor clamp applied identically to the fit.
        a, b, c, d = theta
        mu, sigma = predict((a, b, c, d, sigma_floor), m, s2)
        loss = float(crps_norm(mu, sigma, y).mean())

        # Fail loud on a diverged step (CR-02): a non-finite loss never satisfies the tol break,
        # so without this the loop would burn every iteration and return NaN params to persist.
        if not math.isfinite(loss):
            raise CalibrationError(f"non-finite CRPS loss at iter {t} — diverged fit")

        if loss < best_loss:
            best_loss = loss
            best_theta = theta.copy()

        if abs(prev_loss - loss) < tol:
            logger.debug("fit_stratum converged at iter %d (loss=%.6g)", t, loss)
            break
        prev_loss = loss

    a, b, c, d = (float(v) for v in best_theta)
    # Belt-and-suspenders (CR-02): best_theta is finite by construction (only updated on a finite
    # loss), but the warm-start itself could be non-finite if iters==0 or the very first step
    # diverged — never return NaN/inf params to the money path.
    if not all(math.isfinite(v) for v in (a, b, c, d)):
        raise CalibrationError(f"non-finite EMOS fit params ({a}, {b}, {c}, {d})")
    return a, b, c, d
