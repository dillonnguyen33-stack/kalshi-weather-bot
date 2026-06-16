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

from weatherquant.calibrate.evaluate import (
    baseline_gaussian,
    evaluate_stratum_oos,
    temporal_split,
)


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
    biased ``m`` (so ``baseline_gaussian``'s mean is off by the bias) and under-dispersed
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


def test_baseline_gaussian_ensemble_uses_sample_std():
    """For an ensemble, the baseline is ``(member mean, member sample std, ddof=1)`` (D-11)."""
    members = np.array([[68.0, 70.0, 72.0]])  # one date, 3 members
    mu, sigma = baseline_gaussian(members)
    assert mu.shape == (1,)
    assert np.isclose(mu[0], 70.0)
    # Sample std (ddof=1) of {68,70,72} = 2.0 exactly.
    assert np.isclose(sigma[0], np.std([68.0, 70.0, 72.0], ddof=1))
    assert np.isclose(sigma[0], 2.0)


def test_baseline_gaussian_deterministic_uses_residual_std():
    """A single-member (deterministic) stratum gets a residual-std spread (Open Q#2 / D-11).

    With one member per date the ensemble sample std is undefined/zero, so the baseline spread
    falls back to the forecast-minus-obs residual std supplied by the caller — never σ=0
    (degenerate over-confidence).
    """
    members = np.array([[70.0], [72.0], [74.0]])  # one member per date — deterministic
    resid_std = 4.0
    mu, sigma = baseline_gaussian(members, residual_std=resid_std)
    assert np.allclose(mu, [70.0, 72.0, 74.0])
    # All spreads equal the residual std (no per-date member dispersion to use).
    assert np.allclose(sigma, resid_std)
    assert np.all(sigma > 0.0)


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
