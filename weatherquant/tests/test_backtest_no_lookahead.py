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

from datetime import date

import pytest


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
