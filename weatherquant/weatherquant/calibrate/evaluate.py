"""Out-of-sample validation harness: temporal split + raw-ensemble baseline (criterion 4).

This is Phase 3's SANITY gate (D-10/D-11/D-12) — it confirms EMOS/NGR calibration actually
adds skill over the raw ensemble the system would otherwise price. It is emphatically NOT the
pre-registered Gate-1 proof (Phase 6: walk-forward, v3 head-to-head, block-bootstrap CIs); it
only answers "does the calibrated per-stratum predictive distribution beat the raw-ensemble
baseline on held-out data?".

**Temporal split (D-10, RESEARCH Pitfall 4).** :func:`temporal_split` orders a stratum's
samples by ``target_date`` and partitions into an EARLIER train slice and a STRICTLY-LATER OOS
slice. A random/shuffled split is explicitly wrong here: it leaks future information into the
fit (look-ahead) and is inconsistent with Phase 6's walk-forward philosophy. The split is the
structural no-look-ahead guard.

**Anti-p-hacking + scope (D-12).** The hyperparameters that drive the fit (``N_MIN``, ``KAPPA``,
``SIGMA_FLOOR_F``, the Adam settings) are fixed by principled research defaults — they are NOT
tuned against this OOS slice, and this slice must stay DISJOINT from the period Phase 6 reserves
as its Gate-1 test set. ``trained_through`` (the last train ``target_date``) is recorded so
Phase 6 can audit/re-derive any historical fit from its data cutoff (D-13).

**Raw-ensemble baseline (D-11, RESEARCH §"Raw-ensemble baseline").** The baseline predictive
Gaussian the system would price WITHOUT EMOS is ``(mu = m, sigma = sqrt(s2))`` — the ensemble's
own mean and dispersion, consistent with the population variance ``s2`` the fit consumes. For a
genuinely deterministic stratum (``s2 == 0`` everywhere) the spread falls back to the TRAIN-slice
forecast-minus-obs **residual std** (Open Question #2 / D-11) — a σ that reflects the model's
realized error, never a degenerate ``σ=0`` — computed from train only so the baseline never peeks
at OOS labels. This baseline lives in exactly one place: :func:`evaluate_stratum_oos_aggregated`.

**Evaluation (D-13 payload).** :func:`evaluate_stratum_oos_aggregated` is the single source of
truth: it fits on the train slice (via ``emos.fit_stratum``), reconstructs the calibrated Gaussian
on both slices with the shared ``link.predict``, scores calibrated and baseline OOS Gaussians with
``crps.crps_norm``, and returns ``(crps_train, crps_oos, crps_baseline_oos)`` plus the split
provenance (``trained_through`` / ``oos_from`` / ``n_train`` / ``n_oos``). :func:`evaluate_stratum_oos`
is a thin wrapper that collapses raw members to ``(m, s2)`` and delegates, so the two cannot drift.

Pure NumPy + stdlib only — no scipy/sklearn (the AST guard
``tests/test_no_forbidden_calibration_deps.py`` enforces it).
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
    "temporal_split",
    "evaluate_stratum_oos",
    "evaluate_stratum_oos_aggregated",
    "OOSResult",
]

logger = logging.getLogger(__name__)


def temporal_split(
    target_dates: Sequence[date],
    *,
    oos_fraction: float = 0.3,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Partition sample indices into an EARLIER train slice + STRICTLY-LATER OOS slice (D-10).

    Orders the samples by ``target_date`` (ascending) and assigns the latest
    ``round(n * oos_fraction)`` samples to the OOS slice, the rest to train. NEVER shuffles —
    a random split leaks future information into the fit (look-ahead, RESEARCH Pitfall 4) and
    breaks the no-look-ahead invariant Phase 6's walk-forward depends on (D-10/D-12).

    The returned index arrays point into the ORIGINAL sample order (so the caller's aligned
    ``m``/``s2``/``y`` arrays are sliced consistently), and the split is guaranteed temporal:
    every train ``target_date`` is ``<=`` every OOS ``target_date``. When dates tie across the
    boundary the OOS slice may start on the same calendar date; callers asserting a *strict*
    train-before-OOS boundary should use distinct per-sample dates (the synthetic harness does).

    Args:
        target_dates: one ``date`` per sample (aligned with the stratum's ``m``/``s2``/``y``).
        oos_fraction: fraction of the (date-sorted) samples reserved for the OOS slice.

    Returns:
        ``(train_idx, oos_idx)`` — disjoint index arrays into the original order; together they
        cover every sample. Both are non-empty for a stratum with ``n >= 2`` distinct samples.
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

    ``crps_train``/``crps_oos`` are the calibrated model's mean CRPS on the train/OOS slices;
    ``crps_baseline_oos`` is the raw-ensemble baseline's mean OOS CRPS (the bar to beat, D-11).
    ``trained_through`` is the LAST train ``target_date`` (the data cutoff Phase 6 re-derives
    from, D-13); ``oos_from`` is the FIRST OOS ``target_date`` (``trained_through < oos_from`` is
    the temporal-split guarantee, D-10). ``n_train``/``n_oos`` are the per-slice sample counts.

    **These metrics describe THIS function's own unpooled fit, not the persisted params (WR-05).**
    The three CRPS values come from an EMOS fit on the OOS *train slice* with NO pooling — a
    generic "can a fresh per-stratum fit generalize?" diagnostic. The row written by
    ``persist.store_calibration_params`` stores its own ``crps_train`` recomputed from the
    *persisted* (possibly season-pooled) params, and pairs it with the ``crps_oos`` /
    ``crps_baseline_oos`` from here. So on a persisted row those two held-out numbers describe a
    DIFFERENT (unpooled, train-slice) fit than ``crps_train`` does — do not read the train↔OOS gap
    on a persisted row as the pooled fit's generalization. A pooled-fit OOS score would need the
    parent slice split too (deferred with the Phase-6 walk-forward harness).
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

    A thin convenience wrapper that collapses the raw members to the ``(mean m, population
    variance s2)`` predictor per date — exactly the aggregation ``strata.assemble_pairs_from_rows``
    performs (``s2 = 0`` for a single deterministic member, D-02) — and delegates to
    :func:`evaluate_stratum_oos_aggregated`. Production never holds raw members (they are
    collapsed at assembly), so the aggregated function is the single source of truth for the
    temporal split, the train fit, and the raw-ensemble baseline; keeping the member path a
    delegating wrapper means the two cannot drift (WR-03/WR-04).

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
    # Collapse to (mean, population variance) — ddof=0 so a single deterministic member gives
    # s2=0, matching strata.py and the aggregated baseline's deterministic-fallback trigger.
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

    The CLI assembles the ledger to one ``(m = ensemble mean, s2 = ensemble variance)`` row per
    ``(city, model, lead, target_date)`` — the individual members are already gone. This is the
    aggregated twin of :func:`evaluate_stratum_oos` (which takes raw members): the SAME temporal
    split and the SAME calibrated train fit, but the raw-ensemble BASELINE spread is ``sqrt(s2)``
    (the ensemble's own dispersion, consistent with the population-variance ``s2`` the fit
    consumes) rather than a per-date member sample std. A genuinely deterministic stratum
    (``s2 == 0`` everywhere) falls back to the TRAIN forecast-minus-obs residual std — computed
    from train only so the baseline never peeks at OOS labels (anti-leakage, D-12) — never a
    degenerate ``σ = 0``.

    Args mirror :func:`evaluate_stratum_oos`, with ``ens_members_f`` replaced by the aligned
    ``m``/``s2`` predictor arrays. Raises ``ValueError`` for ``< 2`` samples (via
    :func:`temporal_split`); the caller must pass ``>= 2`` distinct target dates for the split
    to be genuinely temporal rather than positional.
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

    # 5. Raw-ensemble baseline on the OOS slice: μ = m, σ = sqrt(s2) (the ensemble's own spread),
    #    clamped to the σ-floor. A fully deterministic stratum (no ensemble spread anywhere) uses
    #    the TRAIN residual std instead — train-only, so the baseline never peeks at OOS labels
    #    (anti-leakage, D-12). Never σ = 0.
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
