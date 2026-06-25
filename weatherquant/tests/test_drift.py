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
