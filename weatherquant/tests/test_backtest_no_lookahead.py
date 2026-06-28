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

import numpy as np
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
    """Test B (CR-05): when a target_date==day pair exists in the decision month it is preferred."""
    from weatherquant.calibrate.strata import TrainingPair
    from weatherquant.verify import backtest

    day = date(2026, 7, 10)
    pairs = [
        TrainingPair(city="NYC", model="hrrr", lead=7, month=7,
                     target_date=date(2026, 7, 8), m=80.0, s2=9.0, y=80.5),
        TrainingPair(city="NYC", model="hrrr", lead=7, month=7,
                     target_date=day, m=86.0, s2=16.0, y=86.5),  # the decision day's pair
    ]
    m_asof, s2_asof = backtest._v3_arm_raw_ensemble(pairs, day, month=7)
    assert m_asof == pytest.approx(86.0)
    assert s2_asof == pytest.approx(16.0)


def test_v3_arm_month_filtered_mean_when_no_decision_day_pair():
    """Test A (CR-02-for-v3): production-normal path — decision-day pair ABSENT.

    A July (month=7, mean ~85°F) set PLUS a January (month=1, mean ~30°F) set, with NO pair whose
    target_date == the July decision day. The returned mean must be the July-only mean (~85°F), NOT
    the cross-season ~57°F midpoint that averaging across BOTH months would yield.
    """
    from weatherquant.calibrate.strata import TrainingPair
    from weatherquant.verify import backtest

    day = date(2026, 7, 10)  # no pair has this target_date — the production-normal branch
    pairs = [
        # July set (month=7) — the decision month, ~85°F, none on the decision day.
        TrainingPair(city="NYC", model="hrrr", lead=7, month=7,
                     target_date=date(2026, 7, 6), m=84.0, s2=9.0, y=84.5),
        TrainingPair(city="NYC", model="hrrr", lead=7, month=7,
                     target_date=date(2026, 7, 8), m=86.0, s2=11.0, y=86.5),
        # January set (month=1) — a different season, ~30°F; MUST NOT enter the July mean.
        TrainingPair(city="NYC", model="hrrr", lead=1, month=1,
                     target_date=date(2026, 1, 6), m=29.0, s2=4.0, y=29.5),
        TrainingPair(city="NYC", model="hrrr", lead=1, month=1,
                     target_date=date(2026, 1, 8), m=31.0, s2=6.0, y=31.5),
    ]
    m_asof, s2_asof = backtest._v3_arm_raw_ensemble(pairs, day, month=7)
    # The July-only mean of {84, 86} = 85.0 / {9, 11} = 10.0 — NOT the cross-season ~57°F midpoint.
    assert m_asof == pytest.approx(85.0)
    assert s2_asof == pytest.approx(10.0)
    assert 70.0 < m_asof < 100.0  # firmly in July's range, never the all-month midpoint


def test_v3_arm_returns_none_when_decision_month_absent():
    """Test C (CR-02-for-v3): a decision month with no as-of pairs returns None (caller logs it)."""
    from weatherquant.calibrate.strata import TrainingPair
    from weatherquant.verify import backtest

    day = date(2026, 7, 10)  # July decision day
    pairs = [  # only January rows — the July month has NO ensemble.
        TrainingPair(city="NYC", model="hrrr", lead=1, month=1,
                     target_date=date(2026, 1, 6), m=29.0, s2=4.0, y=29.5),
        TrainingPair(city="NYC", model="hrrr", lead=1, month=1,
                     target_date=date(2026, 1, 8), m=31.0, s2=6.0, y=31.5),
    ]
    assert backtest._v3_arm_raw_ensemble(pairs, day, month=7) is None


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
    result = backtest._v3_arm_raw_ensemble(pairs, day, month=7)
    assert result is not None
    _m, s2 = result
    v3_sigma = math.sqrt(s2)
    assert v3_sigma == pytest.approx(5.0)
    # The WQ EMOS sigma is floored/shaped by calibration (>= SIGMA_FLOOR_F, typically small here);
    # the raw ensemble spread of 5.0°F is a DIFFERENT quantity than the WQ blended sigma (CR-05).
    from weatherquant.calibrate.strata import SIGMA_FLOOR_F
    assert v3_sigma != SIGMA_FLOOR_F


def test_point_in_time_bias_is_month_filtered_not_cross_season():
    """WR-01: the v3 point-in-time bias is measured from the DECISION-MONTH pairs only.

    The v3 mean is already month-filtered (``_v3_arm_raw_ensemble(..., month=day.month)``). The bias
    added to it MUST mirror that filter — otherwise ``v3_mu = july_mean + cross_season_bias`` re-
    introduces exactly the cross-season contamination the phase removed from the mean. Here July
    carries a +1.0 residual and January a -1.0 residual: the JULY-only bias is +1.0, while the
    cross-season POOLED bias is 0.0 (the two cancel). The month-filtered helper must return the
    July-only +1.0, never the pooled 0.0.
    """
    from weatherquant.calibrate.strata import TrainingPair
    from weatherquant.verify import backtest

    pairs = [
        # July (month=7): obs runs WARM of the ensemble by +1.0 (y - m = +1.0).
        TrainingPair(city="NYC", model="hrrr", lead=7, month=7,
                     target_date=date(2026, 7, 6), m=84.0, s2=9.0, y=85.0),
        TrainingPair(city="NYC", model="hrrr", lead=7, month=7,
                     target_date=date(2026, 7, 8), m=86.0, s2=9.0, y=87.0),
        # January (month=1): obs runs COLD of the ensemble by -1.0 (y - m = -1.0).
        TrainingPair(city="NYC", model="hrrr", lead=1, month=1,
                     target_date=date(2026, 1, 6), m=30.0, s2=4.0, y=29.0),
        TrainingPair(city="NYC", model="hrrr", lead=1, month=1,
                     target_date=date(2026, 1, 8), m=32.0, s2=4.0, y=31.0),
    ]
    # The pooled bias over ALL months is mean(+1, +1, -1, -1) = 0.0 (the WR-01 contamination value).
    assert backtest._point_in_time_bias(pairs) == pytest.approx(0.0)
    # The month-filtered bias for July is the July-only mean residual = +1.0 (never the pooled 0.0).
    assert backtest._point_in_time_bias_for_month(pairs, month=7) == pytest.approx(1.0)
    # And January's is the January-only residual = -1.0.
    assert backtest._point_in_time_bias_for_month(pairs, month=1) == pytest.approx(-1.0)


def test_point_in_time_bias_for_month_equals_pooled_when_uniform():
    """WR-01: when every month's residual is uniform, the month-filtered bias == the pooled bias.

    This is the invariance the uniform-bias seeded fixtures relied on (the contamination was hidden
    BECAUSE every month had the same +0.5 residual). With a uniform +0.5 residual across two
    seasons, the July-only bias and the pooled bias both equal +0.5 — so the month filter is a no-op
    here, by construction, and the existing uniform-bias tests stay green.
    """
    from weatherquant.calibrate.strata import TrainingPair
    from weatherquant.verify import backtest

    pairs = [
        TrainingPair(city="NYC", model="hrrr", lead=7, month=7,
                     target_date=date(2026, 7, 6), m=84.0, s2=9.0, y=84.5),
        TrainingPair(city="NYC", model="hrrr", lead=7, month=7,
                     target_date=date(2026, 7, 8), m=86.0, s2=9.0, y=86.5),
        TrainingPair(city="NYC", model="hrrr", lead=1, month=1,
                     target_date=date(2026, 1, 6), m=30.0, s2=4.0, y=30.5),
        TrainingPair(city="NYC", model="hrrr", lead=1, month=1,
                     target_date=date(2026, 1, 8), m=32.0, s2=4.0, y=32.5),
    ]
    pooled = backtest._point_in_time_bias(pairs)
    july = backtest._point_in_time_bias_for_month(pairs, month=7)
    assert pooled == pytest.approx(0.5)
    assert july == pytest.approx(pooled)


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


# --- 06-06 Task 4: seeded NON-EMPTY-ledger end-to-end proof that CR-02/CR-04/CR-05 are closed ---
#
# The prior integration tests run against an EMPTY DB, so the non-empty walk_forward scoring path
# (where CR-02/CR-05 live) was never exercised. THIS test seeds a real two-season ledger and is the
# standing proof: a green unit suite alone is NOT the evidence.

from datetime import datetime, timedelta, timezone  # noqa: E402

from weatherquant.calibrate.strata import N_MIN  # noqa: E402 - the season-retention floor (30)
from weatherquant.registry import get_city  # noqa: E402
from weatherquant.time import settlement_window  # noqa: E402

_F_TO_K = lambda f: (f - 32.0) * 5.0 / 9.0 + 273.15  # noqa: E731 - test-only inverse of the K→°F seam

# --- 06-10: PRODUCTION-PATH stamping (GAP 3 / VER-03 — the no-look-ahead PROOF) ----------------
#
# The Gate-1 scored window. D_START is the FIRST replayed July decision day; the window is INSIDE
# July so day.month == 7 selects the July fit on BOTH arms.
_GATE1_START = date(2026, 7, 10)
_GATE1_END = date(2026, 7, 13)  # half-open → scored days 07-10, 07-11, 07-12

# WHY THE EXCLUSION IS WIDER THAN ONE DAY (the feasibility cliff): under realistic stamping a
# July day d's settled obs is available_at = settlement_window(NYC, d).end_utc ≈ 05:00 UTC on d+1.
# The decision cutoff for day D is _decision_cutoff(D) = D-1d 00:00 UTC. A July pair survives the
# as-of training read for D iff settlement_window(NYC, d).end_utc < D-1d 00:00 UTC, i.e. 05:00 UTC
# on d+1 < 00:00 UTC on D-1 ⟺ d <= D-3. So for D_START = 2026-07-10 the surviving JULY days are
# 07-01..07-07 (7 pairs) — a contiguous July block alone is NOT >= N_MIN. JJA pools June+July+Aug
# (season 2), so we seed JUNE 01..30 as well: all 30 June days settle before the 07-10 cutoff, and
# 30 (June) + 7 (July) = 37 surviving JJA pairs >= N_MIN (30) → the July month-fit is RETAINED and
# the day SCORES. (Verified arithmetic, NOT "extend if starved" prose.) The July month-fit also
# requires >= 1 surviving July pair so a July entry exists in by_month — 07-01..07-07 supply 7.
_SEED_JUNE_DAYS = 30  # June 01..30 — JJA-season fill so >= N_MIN JJA pairs survive the cutoff
_SEED_JULY_DAYS = 31  # July 01..31 — the scored month; 07-01..07-07 survive the 07-10 cutoff


def _decision_cutoff_utc(D: date) -> datetime:
    """Mirror backtest._decision_cutoff: D - PRIMARY_DECISION_LEAD_DAYS (=1) at 00:00 UTC."""
    return datetime(D.year, D.month, D.day, tzinfo=timezone.utc) - timedelta(days=1)


def _surviving_july_pair_count(D: date) -> int:
    """The as-of July pair count surviving the cutoff for decision day ``D`` (whitebox check).

    A July day d survives iff settlement_window(NYC, d).end_utc < _decision_cutoff(D). This is the
    explicit count the seeding arithmetic guarantees is >= 1 (a July month entry exists); the JJA
    season count (June+July) is what must clear N_MIN — see ``_surviving_jja_pair_count``.
    """
    city = get_city("NYC")
    cutoff = _decision_cutoff_utc(D)
    return sum(
        1
        for i in range(_SEED_JULY_DAYS)
        if settlement_window(city, date(2026, 7, 1) + timedelta(days=i)).end_utc < cutoff
    )


def _surviving_jja_pair_count(D: date) -> int:
    """The as-of JJA-season (June+July) pair count surviving the cutoff for decision day ``D``.

    This is the count that must be >= N_MIN for the season parent to be retained (and hence the
    July month-fit kept). Asserted in the e2e tests so a future seeding regression that starves the
    fit fails LOUD as an explicit count, not a silent no_month_fit / 0 scored records.
    """
    city = get_city("NYC")
    cutoff = _decision_cutoff_utc(D)
    jja = [date(2026, 6, 1) + timedelta(days=i) for i in range(_SEED_JUNE_DAYS)] + [
        date(2026, 7, 1) + timedelta(days=i) for i in range(_SEED_JULY_DAYS)
    ]
    return sum(1 for d in jja if settlement_window(city, d).end_utc < cutoff)


def _seed_two_season_ledger(conn, member_offsets=(-2.0, 0.0, 2.0)):
    """Seed NYC/hrrr forecasts+observations on the PRODUCTION no-look-ahead path (GAP 3 / VER-03).

    ``member_offsets`` sets each day's ensemble members as ``base_f + offset``; the default
    ``(-2, 0, +2)`` gives the s2=8/3 JJA ensemble. Pass a single offset (e.g. ``(0.0,)``) for a
    DETERMINISTIC single-member ensemble (s2=0) — the CR-1 floor regression.

    Decision-day settled obs are stamped at ``settlement_window(get_city("NYC"), D).end_utc`` (the
    realistic LST settlement instant on/after D), NOT back-dated to 2026-01-01. Forecasts are
    stamped at ``D-2d`` 00:00 UTC — a fixed instant STRICTLY BEFORE the ``D-1d`` 00:00 UTC cutoff,
    so the decision day's OWN forecast is admitted (d-2 < D-1) yet its settled obs is NOT (settling
    ≈ 05:00 UTC on D+1, which is >= the cutoff). Result: under ``available_at < cutoff`` the
    decision-day PAIR is correctly ABSENT (no obs) — the production-normal branch — while the
    outcome read (``< D+2d``) still sees the obs to score o_i.

    Seeds JUNE 01..30 + JULY 01..31 (JJA season) so >= N_MIN JJA pairs survive the 07-10 cutoff and
    the July month-fit is retained (see the module-level feasibility arithmetic). JANUARY 01..31 is
    seeded with the SAME realistic stamping as a cross-season distractor (DJF; its own season is
    below N_MIN and skipped — it never wins the month-filter, it just proves the v3/WQ month-filters
    pick July not a cross-season midpoint). All JJA forecasts share members {83, 85, 87} (s2 = 8/3).
    """
    from weatherquant.ingest.writer import insert_forecast, insert_observation

    def _seed_month(year: int, month: int, base_f: float, n_days: int):
        for i in range(n_days):
            d = date(year, month, 1 + i)
            members_f = [base_f + off for off in member_offsets]  # default {-2,0,+2} → s2 = 8/3
            cycle = datetime(d.year, d.month, d.day, 0, tzinfo=timezone.utc)
            # Forecasts: stamped at D-2d 00:00 UTC — STRICTLY before the D-1d 00:00 UTC cutoff so the
            # decision day's own forecast is NOT dropped (the cycle field stays d 00:00 UTC).
            fc_avail = datetime(d.year, d.month, d.day, tzinfo=timezone.utc) - timedelta(days=2)
            for member, mf in enumerate(members_f):
                insert_forecast(
                    conn, city="NYC", target_date=d, model="hrrr", lead=0, member=member,
                    temp_kelvin=_F_TO_K(mf), cycle=cycle,
                    station_lat=40.779, station_lon=-73.969, grid_distance_m=1000.0,
                    available_at=fc_avail,
                )
            # The verifying daily-high obs: stamped at the realistic LST SETTLEMENT instant on/after
            # day d (≈ 05:00 UTC on d+1) — excluded from the < D-1d training read (no look-ahead),
            # included in the < D+2d outcome read (scores o_i).
            obs_avail = settlement_window(get_city("NYC"), d).end_utc
            insert_observation(
                conn, city="NYC", target_date=d, source="asos", daily_high_f=base_f + 0.5,
                available_at=obs_avail,
            )

    _seed_month(2026, 1, base_f=30.0, n_days=31)  # January, DJF — cross-season distractor
    _seed_month(2026, 6, base_f=85.0, n_days=_SEED_JUNE_DAYS)  # June, JJA — season-retention fill
    _seed_month(2026, 7, base_f=85.0, n_days=_SEED_JULY_DAYS)  # July, JJA — the scored month


@pytest.mark.integration
def test_walk_forward_uses_decision_month_fit_not_all_month_average(pg_conn):
    """CR-02 (seeded e2e, PRODUCTION path): a scored July wq_mu tracks the JULY fit, not all-month.

    06-10: now on the REALISTICALLY-stamped ledger (decision-day obs at settlement, forecasts at
    D-2d). The decision-day pair is ABSENT (no-look-ahead) yet the day STILL scores because >= N_MIN
    JJA pairs survive the cutoff — the green assertion certifies the invariant, not a back-dated
    fixture. An explicit surviving-pair-count assertion fails loud on a starved fit.
    """
    from weatherquant.verify import backtest

    # The seeding arithmetic GUARANTEE, asserted explicitly so a starved fit fails as a count (not a
    # silent no_month_fit / 0-record suite): >= N_MIN JJA pairs survive the first scored day's cutoff.
    assert _surviving_jja_pair_count(_GATE1_START) >= N_MIN, (
        "seeding regression: < N_MIN JJA pairs survive the cutoff → the July fit would be starved"
    )
    assert _surviving_july_pair_count(_GATE1_START) >= 1, "no surviving July pair → no July month entry"

    _seed_two_season_ledger(pg_conn)
    records, coverage = backtest.walk_forward(
        pg_conn, "KXHIGHNY", "hrrr", lead=0,
        start=_GATE1_START, end=_GATE1_END,  # a window INSIDE July
        oos_slice=(date(2025, 1, 1), date(2025, 6, 1)),  # disjoint from the Gate-1 window
    )
    scored = [r for r in records if r.excluded_reason is None]
    assert scored, "the realistically-stamped ledger must STILL produce at least one scored record"
    # No scored decision day may have been dropped for a starved fit (the retention held on the path).
    scored_days = {r.day for r in scored}
    starved = [
        e for e in coverage
        if e.get("day") in scored_days and e.get("reason") in ("no_month_fit", "no_v3_ensemble")
    ]
    assert not starved, f"a scored day must not also be coverage-logged as starved: {starved}"
    # CR-02: the WQ predictive mean tracks the JULY month (~85°F), NOT the cross-month midpoint
    # (~57.5°F that an equal-weight Jan+Jul average would produce).
    wq_mus = [r.wq_mu for r in scored if r.wq_mu is not None]
    assert wq_mus, "scored records must carry wq_mu"
    assert min(wq_mus) > 70.0  # firmly in July's range, far above the ~57.5°F all-month midpoint


@pytest.mark.integration
def test_v3_arm_priced_from_raw_ensemble_spread_not_wq_sigma(pg_conn):
    """CR-05 (seeded e2e, PRODUCTION path): v3_sigma == sqrt(month-mean s2), != wq_sigma.

    06-10: the decision-day pair is now ABSENT (realistic stamping), so the v3 arm takes 06-08's
    MONTH-FILTERED fallback — the mean s2 over the surviving JULY pairs. Each July day's members are
    {83, 85, 87} → population variance 8/3 per pair, and the mean of identical 8/3 values is still
    8/3, so the expected v3_sigma is UNCHANGED at sqrt(8/3) ≈ 1.633 by construction. Only the REASON
    it holds (month-filtered ensemble over absent-decision-day pairs, not a single back-dated
    decision-day triple) changes — do NOT "fix" the still-correct expected value.
    """
    import math

    from weatherquant.verify import backtest

    _seed_two_season_ledger(pg_conn)
    records, _coverage = backtest.walk_forward(
        pg_conn, "KXHIGHNY", "hrrr", lead=0,
        start=_GATE1_START, end=_GATE1_END,
        oos_slice=(date(2025, 1, 1), date(2025, 6, 1)),
    )
    scored = [r for r in records if r.excluded_reason is None]
    assert scored
    # Month-filtered mean of s2 over July pairs, each members {83, 85, 87} → mean s2 = 8/3 → sqrt.
    expected_v3_sigma = math.sqrt(np.var([83.0, 85.0, 87.0]))
    v3_sigmas = {round(r.v3_sigma, 6) for r in scored if r.v3_sigma is not None}
    wq_sigmas = {round(r.wq_sigma, 6) for r in scored if r.wq_sigma is not None}
    assert v3_sigmas, "scored records must carry v3_sigma"
    # CR-05: v3 spread is the raw month-filtered ensemble sqrt(mean s2), independent of wq_sigma.
    assert all(s == pytest.approx(expected_v3_sigma, abs=1e-3) for s in v3_sigmas)
    assert v3_sigmas.isdisjoint(wq_sigmas)  # the two arms' spreads are genuinely different


@pytest.mark.integration
def test_v3_deterministic_ensemble_floors_sigma_so_crps_never_crashes(pg_conn):
    """CR-1: a single-member (deterministic) ensemble has s2==0 → sqrt(s2)==0. The per-record v3
    spread MUST be floored (``_V3_SPREAD_FLOOR``) at the record source, because the CRPS arm
    (``crps_blend``) raises on ``sigma<=0`` with no fallback — an un-floored 0 would abort the whole
    Gate-1 verdict inside the bootstrap resample loop. Before the fix every scored record carried
    ``v3_sigma==0.0`` and ``_crps_score_fn`` raised.
    """
    from weatherquant.cli.verify import _crps_score_fn
    from weatherquant.verify import backtest, metrics
    from weatherquant.verify.v3_reference import _V3_SPREAD_FLOOR

    _seed_two_season_ledger(pg_conn, member_offsets=(0.0,))  # 1 member → s2 = 0 everywhere
    records, _coverage = backtest.walk_forward(
        pg_conn, "KXHIGHNY", "hrrr", lead=0,
        start=_GATE1_START, end=_GATE1_END,
        oos_slice=(date(2025, 1, 1), date(2025, 6, 1)),
    )
    scored = [r for r in records if r.excluded_reason is None]
    assert scored
    v3_sigmas = [r.v3_sigma for r in scored if r.v3_sigma is not None]
    assert v3_sigmas
    # Floored to the v3 floor, NOT the un-floored 0.0 that would crash crps_blend.
    assert all(s == pytest.approx(_V3_SPREAD_FLOOR) for s in v3_sigmas)
    # The REAL scoring path must not raise on the deterministic ensemble (the actual crash site).
    score_fn = _crps_score_fn(scored, metrics)
    score_fn(sorted({r.day for r in scored}))


@pytest.mark.integration
def test_verify_window_must_be_disjoint_from_phase3_oos(pg_conn):
    """CR-04 (seeded e2e): an OOS slice overlapping the Gate-1 window raises on the real path."""
    from weatherquant.verify import backtest

    _seed_two_season_ledger(pg_conn)
    # OOS slice [2026-07-11, 2026-07-20) overlaps the Gate-1 window [2026-07-10, 2026-07-13).
    with pytest.raises(ValueError):
        backtest.walk_forward(
            pg_conn, "KXHIGHNY", "hrrr", lead=0,
            start=_GATE1_START, end=_GATE1_END,
            oos_slice=(date(2026, 7, 11), date(2026, 7, 20)),
        )
    # A disjoint slice scores records (the guard passes, the non-empty path runs) — confirm the
    # realistically-stamped ledger STILL scores under the window-geometry (unaffected) test.
    records, _coverage = backtest.walk_forward(
        pg_conn, "KXHIGHNY", "hrrr", lead=0,
        start=_GATE1_START, end=_GATE1_END,
        oos_slice=(date(2025, 1, 1), date(2025, 6, 1)),
    )
    assert [r for r in records if r.excluded_reason is None]


# --- 06-07 Task 1: CR-03 the ladder tiles (-inf, +inf) with open tail buckets -----------------
#
# A CLOSED ±4σ degree ladder scores any realized high in a tail o_i=0 EVERYWHERE (no bucket is its
# YES) with NO coverage-log entry — a silent drop of exactly the surprise/tail days. Task 1 tiles
# the ladder with a `<= lo` open-lower and a `>= hi` open-upper bucket so EVERY realized high has a
# YES bucket, and coverage-logs a tail-settled day as `tail_settlement` (still scored — an audit
# annotation, not a drop).


def test_ladder_for_day_tiles_open_tail_buckets():
    """CR-03: _ladder_for_day appends an open_lo lower-tail and an open_hi upper-tail bucket."""
    from weatherquant.verify import backtest

    ladder = backtest._ladder_for_day(85.0, 2.0)
    assert ladder, "a finite (mu, sigma) must produce a non-empty ladder"
    open_los = [b for b in ladder if b["edges"][2] is True]  # open_lo flag in (lo, hi, open_lo, ...)
    open_his = [b for b in ladder if b["edges"][3] is True]  # open_hi flag
    assert len(open_los) == 1, "exactly one <= lo open-lower tail bucket"
    assert len(open_his) == 1, "exactly one >= hi open-upper tail bucket"
    # The open spans reach the ∓inf sentinel so the ladder tiles (-inf, +inf).
    lower_span = open_los[0]["span"]
    upper_span = open_his[0]["span"]
    assert lower_span[0] == -math.inf
    assert upper_span[1] == math.inf


def test_tiled_ladder_wq_probs_sum_to_one():
    """CR-03/VER-04: the full tiled WQ ladder (open tails included) sums to ~1 — tiles (-inf, +inf)."""
    from weatherquant.price.buckets import bucket_probs
    from weatherquant.verify import backtest

    mu_b, sigma_b = 85.0, 3.0
    ladder = backtest._ladder_for_day(mu_b, sigma_b)
    wq = bucket_probs(mu_b, sigma_b, [b["span"] for b in ladder])
    assert float(wq.sum()) == pytest.approx(1.0, abs=1e-9)


def test_tail_high_lands_in_open_upper_bucket_not_zero_everywhere():
    """CR-03: a realized high ABOVE the interior range is o_i=1 in the open-upper bucket (not 0 ⁠all)."""
    from weatherquant.verify import backtest

    mu_b, sigma_b = 85.0, 2.0
    ladder = backtest._ladder_for_day(mu_b, sigma_b)
    # A surprise high far above center + 4σ — with a closed ladder it would be o_i=0 everywhere.
    y_tail = mu_b + 100.0
    outcomes = [
        backtest._outcome_for_bucket(y_tail, *b["edges"]) for b in ladder
    ]
    assert sum(outcomes) == 1, "the tail high must land in exactly one (open-upper) bucket"
    yes_bucket = ladder[outcomes.index(1)]
    assert yes_bucket["edges"][3] is True, "the YES bucket for a high tail is the open-upper bucket"


# --- 06-08 Task 2: the v3 arm month-filter is NOT the cross-season midpoint (seeded e2e) -------
#
# GAP 1 / SC2 / VER-04: on the production no-look-ahead path the v3 arm previously averaged m/s2
# across the ENTIRE as-of training set (all seasons), flattening a July baseline toward the
# cross-season ~57°F midpoint (verifier probe: v3_mu=56.27 vs wq_mu=85.5 — voiding the
# apples-to-apples head-to-head). This seeded regression proves the month-filtered v3 arm now
# prices the July range (v3_mu > 70°F). 06-10: re-stamped to the REALISTIC production path — the
# decision-day pair is now ABSENT (no-look-ahead) and the v3 arm takes 06-08's month-filtered
# fallback, so this proves the month-filter on the absent-decision-day-pair PRODUCTION branch.


@pytest.mark.integration
def test_v3_arm_month_filtered_not_cross_season(pg_conn):
    """GAP 1/VER-04 (seeded e2e): every scored July v3_mu tracks July (>70°F), not the ~57°F midpoint."""
    from weatherquant.verify import backtest

    _seed_two_season_ledger(pg_conn)
    records, _coverage = backtest.walk_forward(
        pg_conn, "KXHIGHNY", "hrrr", lead=0,
        start=_GATE1_START, end=_GATE1_END,  # a window INSIDE July
        oos_slice=(date(2025, 1, 1), date(2025, 6, 1)),  # disjoint from the Gate-1 window
    )
    scored = [r for r in records if r.excluded_reason is None]
    assert scored, "the seeded non-empty ledger must produce at least one scored record"
    v3_mus = [r.v3_mu for r in scored if r.v3_mu is not None]
    assert v3_mus, "scored records must carry v3_mu"
    # The v3 arm prices the JULY range (~85°F), NOT the ~57°F cross-season Jan+Jul midpoint that the
    # pre-fix all-month average produced. Every scored July record must be firmly above 70°F.
    assert min(v3_mus) > 70.0


# --- 06-10: the no-look-ahead PROOF on the PRODUCTION path (GAP 3 / VER-03) --------------------
#
# The verifier probe re-stamped decision-day obs realistically and got ZERO scored records — the
# green suite certified a back-dated FIXTURE, not the invariant. These two tests are the inverse:
# with decision-day obs at settlement, walk_forward STILL scores, and the v3 arm is month-correct
# on the PRODUCTION path (the direct closure of the verifier's v3_mu=56.27 probe).


@pytest.mark.integration
def test_no_lookahead_scores_under_realistic_stamping(pg_conn):
    """VER-03 (seeded e2e): decision-day obs at settlement → walk_forward STILL scores >= 1 record.

    The inverse of the verifier probe's "0 scored records". The July window scores under the realistic
    no-look-ahead stamping (decision-day obs excluded from training, present in the outcome read), and
    NO scored day is coverage-logged as a starved fit — the production path ran end-to-end.
    """
    from weatherquant.verify import backtest

    _seed_two_season_ledger(pg_conn)
    records, coverage = backtest.walk_forward(
        pg_conn, "KXHIGHNY", "hrrr", lead=0,
        start=_GATE1_START, end=_GATE1_END,
        oos_slice=(date(2025, 1, 1), date(2025, 6, 1)),
    )
    scored = [r for r in records if r.excluded_reason is None]
    assert scored, "realistic stamping must STILL score (inverse of the verifier's 0-record probe)"
    scored_days = {r.day for r in scored}
    starved = [
        e for e in coverage
        if e.get("day") in scored_days and e.get("reason") in ("no_month_fit", "no_v3_ensemble")
    ]
    assert not starved, f"no scored day may be coverage-logged as a starved fit: {starved}"


@pytest.mark.integration
def test_v3_arm_month_tracked_under_realistic_stamping(pg_conn):
    """VER-04 (seeded e2e): every scored July v3_mu > 70°F on the realistically-stamped PRODUCTION path.

    The direct closure of the verifier's v3_mu=56.27 probe: with decision-day obs stamped at
    settlement (the pair ABSENT), the month-filtered v3 fallback prices July, not the cross-season
    midpoint.
    """
    from weatherquant.verify import backtest

    _seed_two_season_ledger(pg_conn)
    records, _coverage = backtest.walk_forward(
        pg_conn, "KXHIGHNY", "hrrr", lead=0,
        start=_GATE1_START, end=_GATE1_END,
        oos_slice=(date(2025, 1, 1), date(2025, 6, 1)),
    )
    scored = [r for r in records if r.excluded_reason is None]
    assert scored
    v3_mus = [r.v3_mu for r in scored if r.v3_mu is not None]
    assert v3_mus, "scored records must carry v3_mu"
    assert min(v3_mus) > 70.0  # month-correct on the PRODUCTION path (closes the v3_mu=56.27 probe)


# --- 06-10 Task 3: the STANDING no-back-dating guard (GAP 3 stays closed) ----------------------
#
# A structural regression guard: the seeders can NEVER silently regress to the 2026-01-01
# back-dating of decision-day settled obs. Read the seeded observation rows back and assert each
# one's available_at is at/after its OWN settlement window end — so a future edit that re-stamps
# obs before their settlement (re-admitting the day's own outcome into the < D-1d training read)
# fails LOUD here, not as a silently re-contaminated "proof".


@pytest.mark.integration
def test_seeded_decision_day_obs_are_not_back_dated(pg_conn):
    """GAP 3 standing guard: every seeded settled obs is stamped at/after its settlement window end."""
    from weatherquant.db import queries

    _seed_two_season_ledger(pg_conn)
    obs_rows = queries.latest(
        pg_conn, "observations", where={"city": "NYC", "source": "asos"}
    )
    assert obs_rows, "the seeder must produce settled observation rows to guard"
    offenders = []
    for row in obs_rows:
        target_date = row["target_date"]
        available_at = row["available_at"]
        settle_end = settlement_window(get_city("NYC"), target_date).end_utc
        if available_at < settle_end:
            offenders.append((target_date, available_at.isoformat(), settle_end.isoformat()))
    assert not offenders, (
        "decision-day settled obs must be stamped at/after their settlement window end "
        f"(no 2026-01-01 back-dating — GAP 3); back-dated rows: {offenders}"
    )
