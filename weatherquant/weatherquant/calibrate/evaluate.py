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

**Raw-ensemble baseline (D-11, RESEARCH §"Raw-ensemble baseline").**
:func:`baseline_gaussian` returns ``(mean, sample_std)`` over the ensemble members per date
(``ddof=1``) — the predictive Gaussian the system would price WITHOUT EMOS. For a deterministic
single-member stratum the member sample std is degenerate (one member ⇒ std 0), so the spread
falls back to the stratum's forecast-minus-obs **residual std** (Open Question #2 / D-11) — a
σ that reflects the model's realized error, never a degenerate ``σ=0``.

**Evaluation (D-13 payload).** :func:`evaluate_stratum_oos` fits on the train slice (via
``emos.fit_stratum``, reconstructing the calibrated Gaussian on the OOS slice with the shared
``link.predict``), scores both the calibrated and baseline OOS Gaussians with ``crps.crps_norm``,
and returns ``(crps_train, crps_oos, crps_baseline_oos)`` plus the split provenance
(``trained_through`` / ``oos_from`` / ``n_train`` / ``n_oos``) ready for persistence.

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
    "baseline_gaussian",
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


def baseline_gaussian(
    ens_members_f: NDArray[np.float64],
    *,
    residual_std: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The raw-ensemble predictive Gaussian ``(mu, sigma)`` per date (D-11).

    The baseline the system would price WITHOUT EMOS: ``mu`` is the per-date ensemble member
    mean and ``sigma`` the per-date member **sample** std (``ddof=1``) — the raw ensemble's own
    dispersion (RESEARCH §"Raw-ensemble baseline"). For a deterministic single-member stratum
    the member sample std is degenerate (one member ⇒ undefined/0), so the spread falls back to
    ``residual_std`` — the stratum's forecast-minus-obs residual std (Open Question #2 / D-11),
    never a degenerate ``σ=0`` that would make the baseline trivially over-confident.

    Args:
        ens_members_f: ensemble members in °F, shape ``(n_dates, n_members)``.
        residual_std: the deterministic-stratum fallback spread (forecast-minus-obs residual
            std, °F). Required when ``n_members < 2``; ignored otherwise.

    Returns:
        ``(mu, sigma)`` each shape ``(n_dates,)`` — the per-date baseline Gaussian. ``sigma`` is
        clamped to ``>= SIGMA_FLOOR_F`` so CRPS never divides by a degenerate spread.
    """
    members = np.asarray(ens_members_f, dtype=float)
    if members.ndim != 2:
        raise ValueError(
            f"ens_members_f must be 2-D (n_dates, n_members), got shape {members.shape}"
        )
    n_members = members.shape[1]
    mu = members.mean(axis=1)

    if n_members >= 2:
        sigma = members.std(axis=1, ddof=1)  # sample std — the raw ensemble's own dispersion
    else:
        # Deterministic single-member stratum: no per-date dispersion to use — fall back to the
        # caller-supplied residual std (Open Question #2 / D-11). Never σ=0.
        if residual_std is None:
            raise ValueError(
                "deterministic stratum (n_members < 2) needs a residual_std fallback spread "
                "(Open Question #2 / D-11) — refusing a degenerate σ=0 baseline."
            )
        sigma = np.full(mu.shape, float(residual_std))

    # Clamp away degenerate spread so CRPS is well-defined (mirrors the EMOS σ-floor, D-09).
    sigma = np.maximum(sigma, SIGMA_FLOOR_F)
    return mu, sigma


@dataclass(frozen=True)
class OOSResult:
    """The criterion-4 evaluation result for one stratum (ready for the D-13 payload).

    ``crps_train``/``crps_oos`` are the calibrated model's mean CRPS on the train/OOS slices;
    ``crps_baseline_oos`` is the raw-ensemble baseline's mean OOS CRPS (the bar to beat, D-11).
    ``trained_through`` is the LAST train ``target_date`` (the data cutoff Phase 6 re-derives
    from, D-13); ``oos_from`` is the FIRST OOS ``target_date`` (``trained_through < oos_from`` is
    the temporal-split guarantee, D-10). ``n_train``/``n_oos`` are the per-slice sample counts.
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
    """Fit on an earlier slice, evaluate calibrated-vs-baseline OOS CRPS (criterion 4, D-10/D-11).

    Pipeline (all per-stratum, no DB):

    1. **Temporal split (D-10).** :func:`temporal_split` reserves the latest ``oos_fraction`` of
       the date-sorted samples for OOS; the rest train. Never shuffled (no look-ahead).
    2. **Aggregate members** to the ``(mean m, variance s2)`` predictor per date — the same
       collapse ``strata.py`` does for the fit (``s2`` is the population variance; 0 for a
       deterministic single-member stratum ⇒ ``d`` inactive, D-02).
    3. **Fit on train** via :func:`weatherquant.calibrate.emos.fit_stratum` (lstsq warm-start +
       pure-NumPy Adam on mean-CRPS).
    4. **Calibrated OOS CRPS.** Reconstruct the calibrated Gaussian on the OOS slice with the
       shared :func:`weatherquant.calibrate.link.predict` link and score with ``crps_norm``.
    5. **Baseline OOS CRPS.** :func:`baseline_gaussian` over the OOS members (deterministic
       strata use the TRAIN-slice forecast-minus-obs residual std — computed only from train so
       the baseline spread itself does not peek at the OOS labels), scored with ``crps_norm``.

    Returns an :class:`OOSResult`. The criterion-4 success condition the caller asserts is
    ``crps_oos <= crps_baseline_oos`` (calibration must not lose to the raw ensemble OOS).

    Args:
        target_dates: one ``date`` per sample (aligned with ``ens_members_f``/``y``).
        ens_members_f: ensemble members in °F, shape ``(n_dates, n_members)``.
        y: verifying daily-high obs (°F), shape ``(n_dates,)``.
        oos_fraction: fraction of the date-sorted samples held out for OOS.
        sigma_floor: the °F σ-floor passed to the fit + baseline clamp (D-09).
    """
    members = np.asarray(ens_members_f, dtype=float)
    y = np.asarray(y, dtype=float)
    if members.ndim != 2:
        raise ValueError(
            f"ens_members_f must be 2-D (n_dates, n_members), got shape {members.shape}"
        )

    train_idx, oos_idx = temporal_split(target_dates, oos_fraction=oos_fraction)

    # Member collapse to the (mean, variance) predictor — population variance (ddof=0) so a
    # single deterministic member gives s2=0 (matches strata.py; D-02). Independent of split.
    m_all = members.mean(axis=1)
    s2_all = members.var(axis=1)  # population variance over members

    m_tr, s2_tr, y_tr = m_all[train_idx], s2_all[train_idx], y[train_idx]
    m_oos, s2_oos, y_oos = m_all[oos_idx], s2_all[oos_idx], y[oos_idx]

    # 3. Fit on the TRAIN slice only (no OOS leakage).
    a, b, c, d = fit_stratum(m_tr, s2_tr, y_tr, sigma_floor)
    params = (a, b, c, d, sigma_floor)

    # 4. Calibrated CRPS on train (in-sample) and OOS (held-out) via the shared link.
    mu_tr, sig_tr = predict(params, m_tr, s2_tr)
    crps_train = float(crps_norm(mu_tr, sig_tr, y_tr).mean())

    mu_oos, sig_oos = predict(params, m_oos, s2_oos)
    crps_oos = float(crps_norm(mu_oos, sig_oos, y_oos).mean())

    # 5. Raw-ensemble baseline on the OOS slice. For a deterministic stratum the spread falls
    #    back to the TRAIN forecast-minus-obs residual std (computed from train only, so the
    #    baseline spread never peeks at OOS labels — anti-leakage, D-12).
    deterministic = members.shape[1] < 2
    residual_std: float | None = None
    if deterministic:
        resid_tr = y_tr - m_tr
        residual_std = max(float(resid_tr.std(ddof=1)) if len(resid_tr) > 1 else 0.0, sigma_floor)
    base_mu, base_sigma = baseline_gaussian(members[oos_idx], residual_std=residual_std)
    crps_baseline_oos = float(crps_norm(base_mu, base_sigma, y_oos).mean())

    train_dates = [target_dates[i] for i in train_idx]
    oos_dates = [target_dates[i] for i in oos_idx]
    result = OOSResult(
        crps_train=crps_train,
        crps_oos=crps_oos,
        crps_baseline_oos=crps_baseline_oos,
        trained_through=max(train_dates),
        oos_from=min(oos_dates),
        n_train=len(train_idx),
        n_oos=len(oos_idx),
    )
    logger.debug(
        "OOS eval: crps_train=%.4f crps_oos=%.4f crps_baseline_oos=%.4f "
        "trained_through=%s oos_from=%s n_train=%d n_oos=%d",
        result.crps_train,
        result.crps_oos,
        result.crps_baseline_oos,
        result.trained_through,
        result.oos_from,
        result.n_train,
        result.n_oos,
    )
    return result


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
