"""Out-of-sample validation harness: temporal split + raw-ensemble baseline (criterion 4).

Phase 3's SANITY gate (D-10/D-11/D-12): does the calibrated per-stratum distribution beat the
raw-ensemble baseline on held-out data? NOT the pre-registered Gate-1 proof (that is Phase 6).

* Temporal split (D-10): :func:`temporal_split` partitions by ``target_date`` into an EARLIER
  train slice and a STRICTLY-LATER OOS slice — never shuffled, which would leak look-ahead.
* Anti-p-hacking (D-12): fit hyperparameters are fixed research defaults, NOT tuned on this OOS
  slice, which must stay disjoint from Phase 6's Gate-1 test set.
* Raw-ensemble baseline (D-11): the no-EMOS Gaussian is ``(mu = m, sigma = sqrt(s2))``; a fully
  deterministic stratum falls back to the TRAIN-slice residual std (train-only, no OOS peek).
  Lives in one place: :func:`evaluate_stratum_oos_aggregated`.
* :func:`evaluate_stratum_oos` is a thin wrapper collapsing raw members to ``(m, s2)`` and
  delegating, so the two cannot drift.

Pure NumPy + stdlib only — no scipy/sklearn (AST-guard enforced).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

from weatherquant.calibrate.crps import crps_norm
from weatherquant.calibrate.emos import fit_stratum
from weatherquant.calibrate.link import predict
from weatherquant.calibrate.strata import SIGMA_FLOOR_F

__all__ = [
    "OOSResult",
    "evaluate_stratum_oos",
    "evaluate_stratum_oos_aggregated",
    "temporal_split",
]

logger = logging.getLogger(__name__)


def temporal_split(
    target_dates: Sequence[date],
    *,
    oos_fraction: float = 0.3,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Partition sample indices into an EARLIER train slice + STRICTLY-LATER OOS slice (D-10).

    Orders by ``target_date`` and assigns the latest ``round(n * oos_fraction)`` to OOS. NEVER
    shuffles — a random split leaks look-ahead (D-10/D-12). Returned indices point into the
    ORIGINAL order; every train date is ``<=`` every OOS date. On a boundary tie the OOS slice
    may start on the same date; callers needing a strict boundary use distinct per-sample dates.

    Args:
        target_dates: one ``date`` per sample (aligned with the stratum's ``m``/``s2``/``y``).
        oos_fraction: fraction of the date-sorted samples reserved for OOS.

    Returns:
        ``(train_idx, oos_idx)`` — disjoint, covering every sample; both non-empty for ``n >= 2``.
    """
    n = len(target_dates)
    if n < 2:
        raise ValueError(f"temporal_split needs >= 2 samples, got {n}")
    if not 0.0 < oos_fraction < 1.0:
        raise ValueError(f"oos_fraction must be in (0, 1), got {oos_fraction}")

    # Stable sort by date — ties keep their original relative order (deterministic).
    order = np.argsort(np.asarray(target_dates, dtype="datetime64[D]"), kind="stable")
    n_oos = int(round(n * oos_fraction))
    n_oos = max(1, min(n - 1, n_oos))  # both slices non-empty

    train_idx = order[: n - n_oos]
    oos_idx = order[n - n_oos :]
    return train_idx, oos_idx


@dataclass(frozen=True)
class OOSResult:
    """The criterion-4 evaluation result for one stratum (ready for the D-13 payload).

    ``crps_train``/``crps_oos`` are the calibrated model's mean CRPS on train/OOS;
    ``crps_baseline_oos`` is the raw-ensemble baseline's OOS CRPS (the bar to beat, D-11).
    ``trained_through`` (last train date, D-13) ``< oos_from`` (first OOS date) by the D-10
    temporal-split guarantee.

    WR-05 caveat (canonical copy; persist.py cross-references this): these metrics describe THIS
    function's UNPOOLED train-slice fit, not the persisted (possibly season-pooled) params. A
    persisted row pairs its own pooled ``crps_train`` with the ``crps_oos``/``crps_baseline_oos``
    from here, so the held-out pair describes a DIFFERENT fit than ``crps_train`` — do not read the
    train↔OOS gap on a persisted row as the pooled fit's generalization. A true pooled-fit OOS
    score needs the parent slice split too (deferred to the Phase-6 walk-forward harness).
    """

    crps_train: float
    crps_oos: float
    crps_baseline_oos: float
    trained_through: date
    oos_from: date
    n_train: int
    n_oos: int


def evaluate_stratum_oos(
    target_dates: Sequence[date],
    ens_members_f: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    oos_fraction: float = 0.3,
    sigma_floor: float = SIGMA_FLOOR_F,
) -> OOSResult:
    """Criterion-4 OOS evaluation from RAW ensemble members (D-10/D-11).

    Collapses raw members to the ``(mean m, population variance s2)`` predictor per date — the
    same aggregation ``strata.assemble_pairs_from_rows`` performs (D-02) — and delegates to
    :func:`evaluate_stratum_oos_aggregated` (the single source of truth; delegating means the two
    cannot drift, WR-03/WR-04).

    Args:
        target_dates: one ``date`` per sample (aligned with ``ens_members_f``/``y``).
        ens_members_f: ensemble members in °F, shape ``(n_dates, n_members)``.
        y: verifying daily-high obs (°F), shape ``(n_dates,)``.
        oos_fraction: fraction of the date-sorted samples held out for OOS.
        sigma_floor: the °F σ-floor passed to the fit + baseline clamp (D-09).
    """
    members = np.asarray(ens_members_f, dtype=float)
    if members.ndim != 2:
        raise ValueError(
            f"ens_members_f must be 2-D (n_dates, n_members), got shape {members.shape}"
        )
    # Collapse to (mean, population variance) — ddof=0 so a single member gives s2=0, matching
    # strata.py and the aggregated baseline's deterministic-fallback trigger.
    m = members.mean(axis=1)
    s2 = members.var(axis=1)
    return evaluate_stratum_oos_aggregated(
        target_dates, m, s2, np.asarray(y, dtype=float),
        oos_fraction=oos_fraction, sigma_floor=sigma_floor,
    )


def evaluate_stratum_oos_aggregated(
    target_dates: Sequence[date],
    m: NDArray[np.float64],
    s2: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    oos_fraction: float = 0.3,
    sigma_floor: float = SIGMA_FLOOR_F,
) -> OOSResult:
    """Criterion-4 OOS evaluation from PRE-AGGREGATED ``(m, s2)`` predictors (D-10/D-11).

    The aggregated twin of :func:`evaluate_stratum_oos`: same temporal split and train fit, but
    the raw-ensemble BASELINE spread is ``sqrt(s2)`` (the ensemble's own dispersion). A fully
    deterministic stratum (``s2 == 0`` everywhere) falls back to the TRAIN residual std —
    train-only, never an OOS peek (D-12) — never a degenerate ``σ = 0``.

    Args mirror :func:`evaluate_stratum_oos` with ``ens_members_f`` replaced by ``m``/``s2``.
    Raises ``ValueError`` for ``< 2`` samples (via :func:`temporal_split`).
    """
    m = np.asarray(m, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    y = np.asarray(y, dtype=float)

    train_idx, oos_idx = temporal_split(target_dates, oos_fraction=oos_fraction)
    m_tr, s2_tr, y_tr = m[train_idx], s2[train_idx], y[train_idx]
    m_oos, s2_oos, y_oos = m[oos_idx], s2[oos_idx], y[oos_idx]

    # 3. Fit the calibrated link on the TRAIN slice only (no OOS leakage).
    a, b, c, d = fit_stratum(m_tr, s2_tr, y_tr, sigma_floor)
    params = (a, b, c, d, sigma_floor)

    # 4. Calibrated CRPS, in-sample (train) and held-out (OOS), via the shared link.
    mu_tr, sig_tr = predict(params, m_tr, s2_tr)
    crps_train = float(crps_norm(mu_tr, sig_tr, y_tr).mean())
    mu_oos, sig_oos = predict(params, m_oos, s2_oos)
    crps_oos = float(crps_norm(mu_oos, sig_oos, y_oos).mean())

    # 5. Raw-ensemble baseline on OOS: μ = m, σ = sqrt(s2) clamped to the floor. A fully
    #    deterministic stratum uses the TRAIN residual std (train-only, no OOS peek; D-12).
    if bool(np.all(s2 <= 0.0)):
        resid_tr = y_tr - m_tr
        residual_std = max(
            float(resid_tr.std(ddof=1)) if len(resid_tr) > 1 else 0.0, sigma_floor
        )
        base_sigma = np.full(m_oos.shape, residual_std)
    else:
        base_sigma = np.maximum(np.sqrt(np.maximum(s2_oos, 0.0)), sigma_floor)
    crps_baseline_oos = float(crps_norm(m_oos, base_sigma, y_oos).mean())

    train_dates = [target_dates[i] for i in train_idx]
    oos_dates = [target_dates[i] for i in oos_idx]
    return OOSResult(
        crps_train=crps_train,
        crps_oos=crps_oos,
        crps_baseline_oos=crps_baseline_oos,
        trained_through=max(train_dates),
        oos_from=min(oos_dates),
        n_train=len(train_idx),
        n_oos=len(oos_idx),
    )
