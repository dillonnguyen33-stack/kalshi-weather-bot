"""Criterion 4 — the OOS sanity gate: calibrated CRPS beats the raw-ensemble baseline.

This is the held-out validation harness for Phase 3 (D-10/D-11). It is a SANITY gate, not
the pre-registered Gate-1 proof (Phase 6, D-12): it only asserts that EMOS/NGR calibration
adds skill over the raw ensemble the system would otherwise price, on a TEMPORAL out-of-sample
split (train on earlier dates, evaluate on a strictly-later held-out slice — never a random
split, RESEARCH Pitfall 4 / D-10).

The synthetic data is built so the raw ensemble is *mis-specified*: its mean is biased and its
spread mis-dispersed relative to the true predictive law. EMOS therefore has genuine skill to
recover (a debiased mean + a recalibrated spread), so the calibrated OOS CRPS must come in
``<=`` the raw-ensemble baseline OOS CRPS per stratum. This is NOT a tautology: if calibration
were a no-op the biased/mis-dispersed ensemble would still lose, and the assertion would fail.

The split is asserted temporal (train target-dates strictly precede OOS target-dates) so a
look-ahead/leakage regression is caught structurally (D-10 / D-12 anti-p-hacking).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from weatherquant.calibrate.crps import crps_norm
from weatherquant.calibrate.evaluate import (
    evaluate_stratum_oos,
    evaluate_stratum_oos_aggregated,
    temporal_split,
)
from weatherquant.calibrate.strata import SIGMA_FLOOR_F


def _synthetic_misspecified_stratum(
    *,
    seed: int,
    n: int,
    n_members: int,
) -> tuple[
    list[date],
    np.ndarray,  # ens_members_f, shape (n, n_members)
    np.ndarray,  # y (°F)
]:
    """Draw an ensemble stratum whose raw forecast is biased + mis-dispersed.

    The TRUE daily-high law is ``y ~ N(true_mu, true_sigma)`` with ``true_mu = a + b*m``
    where ``m`` is the ensemble mean. The raw ENSEMBLE members, however, are centered at a
    biased ``m`` (so the raw-ensemble baseline mean is off by the bias) and under-dispersed
    (members cluster tighter than the true σ), so the raw-ensemble Gaussian is mis-calibrated.
    EMOS recovers the debias ``(a, b)`` and the spread inflation ``(c, d)`` — genuine skill.
    """
    rng = np.random.default_rng(seed)
    # Per-date latent ensemble-mean forecast (°F).
    m = rng.normal(70.0, 8.0, n)
    # The raw ensemble members scatter tightly (under-dispersed) around m.
    member_spread = 0.8
    members = m[:, None] + rng.normal(0.0, member_spread, size=(n, n_members))
    # True predictive law: a debiasing affine map of m, with a WIDER true spread than the
    # ensemble's own scatter — so the raw ensemble is both biased (a!=0,b!=1) and too sharp.
    true_a, true_b, true_sigma = 3.0, 0.92, 3.5
    true_mu = true_a + true_b * m
    y = rng.normal(true_mu, true_sigma)
    # Ascending target dates (one per sample) so the split is unambiguously temporal.
    base = date(2024, 1, 1)
    dates = [base + timedelta(days=i) for i in range(n)]
    return dates, members, y


def test_temporal_split_orders_by_date_no_shuffle():
    """The split is temporal: every train date strictly precedes every OOS date (D-10)."""
    base = date(2024, 1, 1)
    # Deliberately pass dates OUT of order to prove the split sorts (never trusts input order).
    dates = [base + timedelta(days=i) for i in (5, 0, 9, 2, 7, 1, 8, 3, 6, 4)]
    values = np.arange(10, dtype=float)

    train_idx, oos_idx = temporal_split(dates, oos_fraction=0.3)

    train_dates = [dates[i] for i in train_idx]
    oos_dates = [dates[i] for i in oos_idx]
    assert train_dates, "train slice must be non-empty"
    assert oos_dates, "OOS slice must be non-empty"
    # Strict temporal separation — the latest train date is BEFORE the earliest OOS date.
    assert max(train_dates) < min(oos_dates)
    # No leakage: the two index sets are disjoint and cover all samples.
    assert set(train_idx).isdisjoint(oos_idx)
    assert sorted([*train_idx, *oos_idx]) == list(range(10))
    # The values carried by the indices are untouched (no shuffle of the payload).
    assert np.array_equal(values[train_idx], np.array([values[i] for i in train_idx]))


def test_calibrated_oos_beats_raw_ensemble_baseline():
    """Criterion 4: calibrated OOS CRPS <= raw-ensemble baseline OOS CRPS (temporal split)."""
    dates, members, y = _synthetic_misspecified_stratum(seed=7, n=2000, n_members=20)

    result = evaluate_stratum_oos(dates, members, y, oos_fraction=0.3)

    crps_train, crps_oos, crps_baseline_oos = (
        result.crps_train,
        result.crps_oos,
        result.crps_baseline_oos,
    )
    # The criterion-4 inequality: calibration must not lose to the raw ensemble OOS.
    assert crps_oos <= crps_baseline_oos, (
        f"calibrated OOS CRPS {crps_oos:.4f} did not beat baseline "
        f"{crps_baseline_oos:.4f} — calibration added no skill"
    )
    # And it should add MEANINGFUL skill on this deliberately mis-specified ensemble
    # (guards against a tautological pass where both are ~equal by construction).
    assert crps_oos < crps_baseline_oos * 0.95
    # All three metrics are finite positive numbers ready for persistence.
    for v in (crps_train, crps_oos, crps_baseline_oos):
        assert np.isfinite(v) and v > 0.0


def test_evaluate_stratum_split_is_temporal():
    """The evaluator's own split keeps train dates strictly before OOS dates (D-10/D-12)."""
    dates, members, y = _synthetic_misspecified_stratum(seed=11, n=400, n_members=10)
    result = evaluate_stratum_oos(dates, members, y, oos_fraction=0.25)
    assert result.trained_through < result.oos_from
    # trained_through is the LAST train date (the data cutoff Phase 6 re-derives from, D-13).
    assert result.n_train > 0 and result.n_oos > 0
    assert result.n_train + result.n_oos == len(y)


def test_aggregated_oos_beats_baseline_on_real_variance():
    """The CLI's aggregated (m, s2) path (CR-01): calibrated OOS CRPS beats a sqrt(s2) raw-ensemble
    baseline on a TEMPORAL split. Feeding the real ensemble variance — not a collapsed single
    pseudo-member that forces s2=0 — is what makes this a meaningful, non-deterministic baseline."""
    dates, members, y = _synthetic_misspecified_stratum(seed=7, n=2000, n_members=20)
    m = members.mean(axis=1)
    s2 = members.var(axis=1)  # population variance — matches strata.assemble_pairs_from_rows
    assert np.all(s2 > 0.0), "ensemble must be genuinely dispersed (not the collapsed pseudo-member)"

    result = evaluate_stratum_oos_aggregated(dates, m, s2, y, oos_fraction=0.3)

    assert result.crps_oos <= result.crps_baseline_oos
    assert result.crps_oos < result.crps_baseline_oos * 0.95  # meaningful skill, not a tautology
    for v in (result.crps_train, result.crps_oos, result.crps_baseline_oos):
        assert np.isfinite(v) and v > 0.0
    # Real distinct dates ⇒ a genuine temporal split, not a positional one (CR-02 / D-10).
    assert result.trained_through < result.oos_from
    assert result.n_train + result.n_oos == len(y)


def test_aggregated_baseline_uses_sqrt_s2_not_deterministic_collapse():
    """Pinpoint CR-01: the aggregated baseline spread is sqrt(s2) (the real ensemble dispersion),
    proving the CLI no longer collapses every stratum to a single pseudo-member (which forced
    s2=0 and the deterministic residual-std branch for everyone)."""
    dates = [date(2024, 1, d) for d in (1, 2, 3, 4)]
    m = np.array([70.0, 71.0, 69.0, 72.0])
    s2 = np.array([9.0, 9.0, 9.0, 9.0])  # ensemble variance 9 ⇒ baseline σ = 3 (≫ σ-floor)
    y = np.array([71.0, 70.0, 70.0, 73.0])

    result = evaluate_stratum_oos_aggregated(dates, m, s2, y, oos_fraction=0.5)

    # Recompute the expected baseline CRPS directly from μ = m_oos, σ = max(sqrt(s2_oos), floor).
    _, oos_idx = temporal_split(dates, oos_fraction=0.5)
    base_sigma = np.maximum(np.sqrt(s2[oos_idx]), SIGMA_FLOOR_F)
    expected = float(crps_norm(m[oos_idx], base_sigma, y[oos_idx]).mean())
    assert result.crps_baseline_oos == pytest.approx(expected)
    # The σ=3 baseline is well above the floor — it is NOT the degenerate deterministic fallback.
    assert base_sigma[0] == pytest.approx(3.0)


def test_aggregated_deterministic_stratum_uses_train_residual_std():
    """A fully deterministic stratum (s2==0 everywhere) falls back to the TRAIN residual std for
    the baseline spread — never σ=0, and train-only so the baseline never peeks at OOS labels
    (D-11/D-12). This is the single deterministic-baseline path now that the raw-member
    baseline_gaussian is gone (WR-03/WR-04)."""
    dates = [date(2024, 1, d) for d in (1, 2, 3, 4)]
    m = np.array([70.0, 71.0, 72.0, 73.0])
    s2 = np.zeros(4)  # deterministic — no ensemble dispersion anywhere
    y = np.array([72.0, 70.0, 75.0, 71.0])

    result = evaluate_stratum_oos_aggregated(dates, m, s2, y, oos_fraction=0.5)

    train_idx, oos_idx = temporal_split(dates, oos_fraction=0.5)
    resid_tr = y[train_idx] - m[train_idx]
    expected_sigma = max(float(resid_tr.std(ddof=1)), SIGMA_FLOOR_F)
    expected = float(
        crps_norm(m[oos_idx], np.full(oos_idx.shape, expected_sigma), y[oos_idx]).mean()
    )
    assert result.crps_baseline_oos == pytest.approx(expected)
    # Residual std (≈2.12) dominates the floor here — proving the fallback fired, not the clamp.
    assert expected_sigma > SIGMA_FLOOR_F


def test_member_wrapper_collapses_to_aggregated_deterministic():
    """The raw-member wrapper collapses a single-member ensemble to s2=0 and delegates to the
    aggregated deterministic path — proving the two paths cannot drift (WR-03)."""
    dates = [date(2024, 1, d) for d in (1, 2, 3, 4)]
    members = np.array([[70.0], [71.0], [72.0], [73.0]])  # one member per date — deterministic
    y = np.array([72.0, 70.0, 75.0, 71.0])

    via_members = evaluate_stratum_oos(dates, members, y, oos_fraction=0.5)
    via_aggregated = evaluate_stratum_oos_aggregated(
        dates, members[:, 0], np.zeros(4), y, oos_fraction=0.5
    )
    assert via_members.crps_baseline_oos == pytest.approx(via_aggregated.crps_baseline_oos)
    assert via_members.crps_oos == pytest.approx(via_aggregated.crps_oos)
