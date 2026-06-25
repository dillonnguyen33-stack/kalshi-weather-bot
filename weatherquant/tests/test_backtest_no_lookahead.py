"""RED contract for the as-of-correct walk-forward backtest (VER-03 / D-09 / D-12).

Three invariants the Wave-3 ``walk_forward`` must satisfy:

* No look-ahead — only ledger rows with ``available_at < cutoff`` are consumed (a row stamped at
  or after the cutoff is EXCLUDED).
* Gate-1 test window asserted DISJOINT from the Phase-3 OOS slice (never scored on tuning data,
  D-12).
* Voided / missing days appear in the coverage log with a reason, never silently dropped (D-09).

The DB-backed assembly parts are ``integration``-marked (need a populated ledger); the as-of-filter
and the OOS-disjointness assertions are kept DB-free where possible. Imports are deferred so
collection stays green while the implementation is RED.
"""

from __future__ import annotations

import math
from datetime import date

import pytest


def _two_month_pairs():
    """Build TrainingPairs across TWO seasons with clearly different means (CR-02 fixture).

    January (DJF, mean≈30°F) and July (JJA, mean≈85°F): each season is its own parent so both
    month-fits are retained (≥N_MIN per season). The means are far apart so an equal-weight
    all-month average (≈57.5) would land BETWEEN them — distinguishable from either month-fit.
    """
    from weatherquant.calibrate.strata import TrainingPair

    pairs: list[TrainingPair] = []
    for i in range(40):  # >= N_MIN (30) per season so each season parent is retained
        # January: cold; obs tracks the ensemble mean (b≈1, small spread).
        m_jan = 30.0 + (i % 5)
        pairs.append(
            TrainingPair(
                city="NYC", model="hrrr", lead=1, month=1,
                target_date=date(2026, 1, 1 + (i % 28)), m=m_jan, s2=4.0, y=m_jan + 0.2,
            )
        )
        # July: hot; same well-behaved relationship at a far-away mean.
        m_jul = 85.0 + (i % 5)
        pairs.append(
            TrainingPair(
                city="NYC", model="hrrr", lead=7, month=7,
                target_date=date(2026, 7, 1 + (i % 28)), m=m_jul, s2=4.0, y=m_jul + 0.2,
            )
        )
    return pairs


def test_blend_arm_selects_decision_month_fit_not_all_month_average():
    """CR-02: _blend_arm_for_day(month=7) tracks the JULY fit, NOT the cross-month midpoint."""
    from weatherquant.verify import backtest

    pairs = _two_month_pairs()
    jan = backtest._blend_arm_for_day(pairs, city_key="NYC", model="hrrr", lead=1, month=1)
    jul = backtest._blend_arm_for_day(pairs, city_key="NYC", model="hrrr", lead=7, month=7)
    assert jan is not None and jul is not None
    mu_jan, _ = jan
    mu_jul, _ = jul
    # The two month-fits price near their own months' means (~32 and ~87), far apart.
    assert mu_jul > 70.0  # July fit is hot
    assert mu_jan < 50.0  # January fit is cold
    # The cross-month equal-weight midpoint would be ~ (mu_jan + mu_jul)/2 — the July fit must
    # NOT collapse toward it (CR-02: never an all-month average).
    midpoint = (mu_jan + mu_jul) / 2.0
    assert mu_jul > midpoint + 10.0


def test_blend_arm_returns_none_for_absent_month():
    """CR-02/D-09: a decision month with no retained month-fit returns None (caller logs it)."""
    from weatherquant.verify import backtest

    pairs = _two_month_pairs()  # only months 1 and 7 are present
    # April (month=4, MAM season) has no pairs → no season parent → no month-fit → None.
    assert backtest._blend_arm_for_day(pairs, city_key="NYC", model="hrrr", lead=4, month=4) is None


def test_v3_arm_uses_raw_decision_day_ensemble_pair():
    """CR-05: _v3_arm_raw_ensemble returns the raw decision-day (m, s2) — prefers target_date==day."""
    from weatherquant.calibrate.strata import TrainingPair
    from weatherquant.verify import backtest

    day = date(2026, 7, 10)
    pairs = [
        TrainingPair(city="NYC", model="hrrr", lead=7, month=7,
                     target_date=date(2026, 7, 8), m=80.0, s2=9.0, y=80.5),
        TrainingPair(city="NYC", model="hrrr", lead=7, month=7,
                     target_date=day, m=86.0, s2=16.0, y=86.5),  # the decision day's pair
    ]
    m_asof, s2_asof = backtest._v3_arm_raw_ensemble(pairs, day)
    assert m_asof == pytest.approx(86.0)
    assert s2_asof == pytest.approx(16.0)


def test_v3_arm_falls_back_to_as_of_mean_when_no_decision_day_pair():
    """CR-05: with no target_date==day pair, fall back to the mean (m, s2) across as-of pairs."""
    from weatherquant.calibrate.strata import TrainingPair
    from weatherquant.verify import backtest

    day = date(2026, 7, 10)
    pairs = [
        TrainingPair(city="NYC", model="hrrr", lead=7, month=7,
                     target_date=date(2026, 7, 6), m=80.0, s2=9.0, y=80.5),
        TrainingPair(city="NYC", model="hrrr", lead=7, month=7,
                     target_date=date(2026, 7, 8), m=84.0, s2=11.0, y=84.5),
    ]
    m_asof, s2_asof = backtest._v3_arm_raw_ensemble(pairs, day)
    assert m_asof == pytest.approx(82.0)
    assert s2_asof == pytest.approx(10.0)


def test_v3_spread_is_sqrt_s2_distinct_from_wq_sigma():
    """CR-05: the v3 spread is sqrt(s2_asof) (raw ensemble), independent of the WQ blended sigma."""
    from weatherquant.calibrate.strata import TrainingPair
    from weatherquant.verify import backtest

    day = date(2026, 7, 10)
    s2_asof = 25.0
    pairs = [
        TrainingPair(city="NYC", model="hrrr", lead=7, month=7,
                     target_date=day, m=86.0, s2=s2_asof, y=86.5),
    ]
    _m, s2 = backtest._v3_arm_raw_ensemble(pairs, day)
    v3_sigma = math.sqrt(s2)
    assert v3_sigma == pytest.approx(5.0)
    # The WQ EMOS sigma is floored/shaped by calibration (>= SIGMA_FLOOR_F, typically small here);
    # the raw ensemble spread of 5.0°F is a DIFFERENT quantity than the WQ blended sigma (CR-05).
    from weatherquant.calibrate.strata import SIGMA_FLOOR_F
    assert v3_sigma != SIGMA_FLOOR_F


def test_paired_record_carries_predictive_params_for_crps():
    """Task 1/2: PairedRecord exposes wq_mu/wq_sigma/y (and v3_mu/v3_sigma) for Plan 06-07 CRPS."""
    from dataclasses import fields

    from weatherquant.verify.backtest import PairedRecord

    names = {f.name for f in fields(PairedRecord)}
    assert {"wq_mu", "wq_sigma", "y", "v3_mu", "v3_sigma"} <= names
    # Defaults keep the existing constructor calls valid.
    rec = PairedRecord(
        day=date(2026, 7, 1), city="KXHIGHNY", bucket=(85, 86), wq_prob=0.6, v3_prob=0.55, o_i=1
    )
    assert rec.wq_mu is None and rec.wq_sigma is None and rec.y is None
    assert rec.v3_mu is None and rec.v3_sigma is None


def test_paired_record_is_a_frozen_coverage_aware_dataclass():
    """PairedRecord exists now (frozen) and carries the D-09 excluded_reason coverage slot."""
    from dataclasses import FrozenInstanceError, fields

    from weatherquant.verify.backtest import PairedRecord

    names = {f.name for f in fields(PairedRecord)}
    assert {"day", "city", "bucket", "wq_prob", "v3_prob", "o_i", "excluded_reason"} <= names
    rec = PairedRecord(
        day=date(2026, 6, 1), city="KXHIGHNY", bucket=(70, 71), wq_prob=0.6, v3_prob=0.55, o_i=1
    )
    assert rec.excluded_reason is None
    with pytest.raises(FrozenInstanceError):
        rec.o_i = 0  # type: ignore[misc]  # frozen — must not be mutable


def test_walk_forward_overlapping_oos_window_fails_loud_d12():
    """A test window overlapping the Phase-3 OOS slice must raise (never score on tuning data)."""
    from weatherquant.verify import backtest

    start, end = date(2026, 1, 1), date(2026, 3, 1)
    overlapping_oos = (date(2026, 2, 1), date(2026, 4, 1))  # overlaps [start, end)
    with pytest.raises((ValueError, AssertionError)):
        backtest.walk_forward(
            None, "KXHIGHNY", "blend", lead=1, start=start, end=end, oos_slice=overlapping_oos
        )


@pytest.mark.integration
def test_walk_forward_excludes_rows_available_at_or_after_cutoff(pg_conn):
    """As-of filter: a forecast stamped >= cutoff is excluded from that day's paired record (D-12)."""
    from weatherquant.verify import backtest

    # The populated-ledger assembly is exercised here; the implementation lands Wave 3 (RED now).
    records, coverage = backtest.walk_forward(
        pg_conn, "KXHIGHNY", "blend", lead=1,
        start=date(2026, 1, 1), end=date(2026, 2, 1), oos_slice=(date(2025, 1, 1), date(2025, 6, 1)),
    )
    # No record may be built from a row that was not yet available at its day's cutoff.
    assert all(r.excluded_reason != "look_ahead" for r in records)


@pytest.mark.integration
def test_walk_forward_logs_voided_days_with_a_reason(pg_conn):
    """Voided/missing settlement days appear in the coverage log with a reason (D-09), not dropped."""
    from weatherquant.verify import backtest

    records, coverage = backtest.walk_forward(
        pg_conn, "KXHIGHNY", "blend", lead=1,
        start=date(2026, 1, 1), end=date(2026, 2, 1), oos_slice=(date(2025, 1, 1), date(2025, 6, 1)),
    )
    assert all("reason" in entry for entry in coverage)
