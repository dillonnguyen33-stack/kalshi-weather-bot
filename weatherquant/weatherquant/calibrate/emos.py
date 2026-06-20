"""Per-stratum EMOS/NGR fit: lstsq warm-start + pure-NumPy Adam (CAL-01, D-06).

Fits the four NGR params ``(a, b, c, d)`` minimizing mean Gaussian CRPS under the D-02 link
(``mu = a + b*m``; ``sigma^2 = max(sigma_floor^2, c^2 + d^2*s2)``). The fit is small and
near-convex, so it stays pure NumPy — no scipy/sklearn (AST-guard enforced); ``np.linalg.lstsq``
is the only allowed linear-algebra helper.

1. Warm-start (D-06): ``(a0, b0)`` from ``np.linalg.lstsq`` on ``[1, m]``; variance params from
   sample moments (``c0 = max(residual.std(), sigma_floor)`` so most strata start above the floor).
2. Adam (D-06): hand-written, using ``link.param_grads`` for the analytic gradient and
   ``crps.crps_norm`` (via ``predict``) for the loss. The σ-floor clamp + variance-gradient mask
   live inside ``predict``/``param_grads`` (Pitfall 1).
3. Convergence: break on ``abs(prev_loss - loss) < tol`` or ``iters``, returning the BEST-loss
   ``theta`` (not the last — a late overshoot must never ship a worse fit, WR-02).
4. Fail loud (CR-02): a non-finite loss or returned params raises ``CalibrationError`` rather
   than handing NaN to ``persist``.

Deterministic models (``s2 == 0``): ``d``'s gradient is identically 0, so ``d`` stays at
``D0_INIT`` and ``sigma^2 = c^2`` — expected, not a bug (D-02). Adam hyperparameters are
principled research defaults, CLI-overridable (ASSUMPTIONS, D-12; see docs/DECISIONS.md).
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
    "ADAM_B1",
    "ADAM_B2",
    "ADAM_EPS",
    "ADAM_ITERS",
    "ADAM_LR",
    "ADAM_TOL",
    "D0_INIT",
    "fit_stratum",
]

logger = logging.getLogger(__name__)

# Adam defaults: principled research ASSUMPTIONS, CLI-overridable via keyword args (D-12).
ADAM_LR: float = 0.05
ADAM_B1: float = 0.9
ADAM_B2: float = 0.999
ADAM_EPS: float = 1e-8
ADAM_ITERS: int = 2000
ADAM_TOL: float = 1e-7

# Small non-zero init for the spread param d; for deterministic models (s2 == 0) its gradient
# is 0 so it stays here — a constant so the deterministic-model test can assert d is unchanged.
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
        m: per-sample ensemble-mean forecast (°F).
        s2: per-sample ensemble variance; all-zeros for a deterministic model ⇒ ``d`` inactive.
        y: verifying daily-high observations (°F).
        sigma_floor: the °F σ-floor (D-09), enforced inside ``predict``/``param_grads``.
        lr, b1, b2, eps, iters, tol: Adam hyperparameters (module-constant defaults).

    Returns:
        ``(a, b, c, d)`` floats. Signs of ``c, d`` are free (only their squares enter); for
        ``s2 == 0`` ``d`` stays at ``D0_INIT`` (D-02 — expected, not a bug).
    """
    m = np.asarray(m, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    y = np.asarray(y, dtype=float)

    # Empty stratum: CRPS mean is 0/0. Fail loud rather than return NaN params (IN-02).
    if len(y) == 0:
        raise CalibrationError("cannot fit an empty stratum (n=0)")

    # 1. lstsq warm-start for the mean params (D-06): regress y on [1, m].
    design = np.column_stack([np.ones_like(m), m])
    (a0, b0), *_ = np.linalg.lstsq(design, y, rcond=None)

    # Variance params from sample moments: c0^2 >= residual variance keeps most strata above
    # the floor (Pitfall 1); d0 small (inactive for deterministic models).
    resid = y - (a0 + b0 * m)
    c0 = max(float(resid.std()), sigma_floor)
    d0 = D0_INIT

    theta = np.array([a0, b0, c0, d0], dtype=float)

    # 2. Pure-NumPy Adam on mean-CRPS. Track BEST-loss theta so a late overshoot never ships a
    #    worse fit than an earlier (or the warm-start) step (WR-02).
    mt = np.zeros(4, dtype=float)
    vt = np.zeros(4, dtype=float)
    prev_loss = np.inf
    best_loss = np.inf
    best_theta = theta.copy()

    for t in range(1, iters + 1):
        a, b, c, d = theta
        g = param_grads(a, b, c, d, sigma_floor, m, s2, y)  # analytic, floor-masked

        mt = b1 * mt + (1.0 - b1) * g
        vt = b2 * vt + (1.0 - b2) * g * g
        mhat = mt / (1.0 - b1**t)
        vhat = vt / (1.0 - b2**t)
        theta = theta - lr * mhat / (np.sqrt(vhat) + eps)

        # Loss via the shared predict() link — same σ-floor clamp as the fit.
        a, b, c, d = theta
        mu, sigma = predict((a, b, c, d, sigma_floor), m, s2)
        loss = float(crps_norm(mu, sigma, y).mean())

        # Fail loud on a diverged step (CR-02): a non-finite loss never satisfies the tol break.
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
    # Belt-and-suspenders (CR-02): best_theta could be non-finite if iters==0 or the first step
    # diverged — never return NaN/inf params to the money path.
    if not all(math.isfinite(v) for v in (a, b, c, d)):
        raise CalibrationError(f"non-finite EMOS fit params ({a}, {b}, {c}, {d})")
    return a, b, c, d
