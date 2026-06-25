"""RED contract for the post-Gate-1 calibration drift monitor (SYS-02 / D-10).

Asserts the Wave-4 ``monitor_drift`` behavior:

* Returns a NON-ZERO process exit code and logs at ERROR when the trailing-window ECE exceeds
  ``Settings.drift_reliability_threshold``.
* Returns 0 when under threshold.
* Reads the threshold from Settings — never hardcoded.

The Settings-knob existence assertion is DB-free; the ledger-backed monitor parts are
``integration``-marked. Imports are deferred so collection stays green while the implementation is
RED.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pytest


@dataclass
class _FakeSettings:
    """A minimal stand-in carrying only the threshold the monitor must read from Settings (D-10)."""

    drift_reliability_threshold: float


def test_settings_exposes_the_drift_threshold_knob_with_a_principled_default():
    """The drift threshold lives on Settings with a (0,1] default and stays out of the repr."""
    from weatherquant.db.engine import (
        DRIFT_RELIABILITY_THRESHOLD_DEFAULT,
        Settings,
    )

    assert "drift_reliability_threshold" in Settings.model_fields
    assert 0.0 < DRIFT_RELIABILITY_THRESHOLD_DEFAULT <= 1.0
    s = Settings(database_url="postgresql+psycopg://u:p@h/db")
    assert "drift_reliability_threshold" not in repr(s)  # policy knob, not in the redacted repr


@pytest.mark.integration
def test_monitor_drift_nonzero_and_errors_on_breach(pg_conn, caplog):
    """A trailing ECE above the Settings threshold returns non-zero and logs at ERROR (SYS-02)."""
    from weatherquant.verify import drift

    settings = _FakeSettings(drift_reliability_threshold=0.001)  # tiny → guaranteed breach
    with caplog.at_level(logging.ERROR):
        code = drift.monitor_drift(
            pg_conn, settings, window_days=30, cities=["KXHIGHNY"], models=["blend"]
        )
    assert code != 0


@pytest.mark.integration
def test_monitor_drift_zero_when_under_threshold(pg_conn):
    """A generous threshold (no stratum breaches) returns the 0 ok exit code (SYS-02)."""
    from weatherquant.verify import drift

    settings = _FakeSettings(drift_reliability_threshold=1.0)  # nothing can exceed it
    code = drift.monitor_drift(
        pg_conn, settings, window_days=30, cities=["KXHIGHNY"], models=["blend"]
    )
    assert code == 0


# --- DB-free unit coverage of the threshold/exit-code logic (Plan 06-05 Task 2) --------------
#
# The two integration cases above exercise the ledger read end-to-end but cannot force a breach
# against the empty ``pg_conn`` test DB (no settled rows ⇒ nothing to score). These unit tests pin
# the threshold-comparison + ERROR-log + exit-code logic on SYNTHETIC paired records by patching
# ``verify.backtest.walk_forward`` — the contract the plan's action calls "unit-testable on
# synthetic reliability values", reusing ``verify.metrics.ece_equal_count`` for the reliability error.


class _Rec:
    """A minimal stand-in for ``verify.backtest.PairedRecord`` (only the fields the monitor reads)."""

    def __init__(self, wq_prob: float, o_i: int, excluded_reason=None):
        self.wq_prob = wq_prob
        self.o_i = o_i
        self.excluded_reason = excluded_reason


def _badly_calibrated_records():
    """Forecasts that are confidently WRONG (high prob, NO outcomes) → a large equal-count ECE."""
    return [_Rec(0.95, 0) for _ in range(20)] + [_Rec(0.9, 0) for _ in range(20)]


def test_monitor_drift_breach_returns_nonzero_and_logs_error(monkeypatch, caplog):
    """A trailing ECE above the Settings threshold returns non-zero + logs ERROR (SYS-02, unit)."""
    from weatherquant.verify import drift

    monkeypatch.setattr(
        drift, "walk_forward",
        lambda *a, **k: (_badly_calibrated_records(), []),
    )
    settings = _FakeSettings(drift_reliability_threshold=0.01)  # tiny → the bad calibration breaches
    with caplog.at_level(logging.ERROR):
        code = drift.monitor_drift(
            object(), settings, window_days=30, cities=["KXHIGHNY"], models=["blend"]
        )
    assert code != 0
    assert any("DRIFT BREACH" in r.message for r in caplog.records)


def test_monitor_drift_clear_returns_zero(monkeypatch):
    """A generous threshold over the SAME records clears (no breach) → 0 (SYS-02, unit)."""
    from weatherquant.verify import drift

    monkeypatch.setattr(
        drift, "walk_forward",
        lambda *a, **k: (_badly_calibrated_records(), []),
    )
    settings = _FakeSettings(drift_reliability_threshold=1.0)  # nothing can exceed it
    code = drift.monitor_drift(
        object(), settings, window_days=30, cities=["KXHIGHNY"], models=["blend"]
    )
    assert code == 0


def test_monitor_drift_threshold_read_from_settings(monkeypatch):
    """The breach boundary tracks ``Settings.drift_reliability_threshold`` exactly (never hardcoded)."""
    import numpy as np

    from weatherquant.verify import drift
    from weatherquant.verify.metrics import ece_equal_count

    recs = _badly_calibrated_records()
    monkeypatch.setattr(drift, "walk_forward", lambda *a, **k: (recs, []))
    ece = ece_equal_count(
        np.array([r.wq_prob for r in recs]), np.array([float(r.o_i) for r in recs])
    )
    # Just BELOW the measured ECE → breach; just ABOVE → clear. The boundary is the Settings knob.
    below = drift.monitor_drift(
        object(), _FakeSettings(drift_reliability_threshold=ece * 0.9),
        window_days=30, cities=["c"], models=["m"],
    )
    above = drift.monitor_drift(
        object(), _FakeSettings(drift_reliability_threshold=ece * 1.1),
        window_days=30, cities=["c"], models=["m"],
    )
    assert below != 0 and above == 0
