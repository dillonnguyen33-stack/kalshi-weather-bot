"""RED tests for the fixed-offset settlement window (TIME-01, TIME-02).

These import ``weatherquant.time``, which does not exist yet (delivered by plan 01-02),
so this module FAILS at collection/import time on purpose — that ImportError is the RED
signal proving the tests gate real behavior (Nyquist Dimension 8).

Behavior under test (RESEARCH Pattern 1, D-03, Pitfall 3):
* ``settlement_window(city, day)`` returns a ``SettlementWindow``.
* ``start_utc`` is tz-aware UTC and equals local-standard-midnight expressed in UTC:
  ``start_utc.hour == (-std_offset_hours) % 24``.
* ``end_utc == start_utc + 24h`` and is EXCLUSIVE (half-open window).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import pytest


def _import_time():
    """Deferred import so collection succeeds and the RED signal is a per-test
    ModuleNotFoundError (weatherquant.time lands in plan 01-02)."""
    from weatherquant.time import SettlementWindow, settlement_window

    return SettlementWindow, settlement_window


@dataclass(frozen=True)
class _StubCity:
    """Minimal stand-in so test_time does not depend on the registry (01-02) too."""

    std_offset_hours: int
    cli_station: str


# Representative offsets across the 7 cities: ET -5, CT -6, MT -7, PT -8.
@pytest.mark.parametrize(
    "std_offset_hours,station",
    [(-5, "KNYC"), (-6, "KMDW"), (-7, "KDEN"), (-8, "KLAX")],
)
def test_start_utc_is_local_standard_midnight(std_offset_hours, station):
    SettlementWindow, settlement_window = _import_time()
    city = _StubCity(std_offset_hours=std_offset_hours, cli_station=station)
    day = date(2025, 1, 15)
    w = settlement_window(city, day)

    assert isinstance(w, SettlementWindow)
    # tz-aware UTC
    assert w.start_utc.tzinfo is not None
    assert w.start_utc.utcoffset() == timedelta(0)
    # local-standard midnight in UTC: hour == (-offset) % 24 (Pitfall 3 sign check)
    assert w.start_utc.hour == (-std_offset_hours) % 24
    # the UTC instant equals local_midnight - offset
    expected_start = datetime(2025, 1, 15, tzinfo=timezone.utc) - timedelta(
        hours=std_offset_hours
    )
    assert w.start_utc == expected_start


@pytest.mark.parametrize("std_offset_hours", [-5, -6, -7, -8])
def test_window_is_24h_half_open(std_offset_hours):
    _SettlementWindow, settlement_window = _import_time()
    city = _StubCity(std_offset_hours=std_offset_hours, cli_station="KXXX")
    w = settlement_window(city, date(2025, 1, 15))
    # exactly 24h because there is NO DST in this window (that is the whole point)
    assert w.end_utc - w.start_utc == timedelta(days=1)
    # end is exclusive: the instant end_utc belongs to the NEXT day's window
    next_day = settlement_window(city, date(2025, 1, 16))
    assert w.end_utc == next_day.start_utc


def test_summer_dst_day_uses_same_fixed_offset(tmp_path=None):
    """TIME-02: during civil DST the standard-offset window appears shifted on the
    civil clock but the fixed std_offset is still what defines the UTC window."""
    _SettlementWindow, settlement_window = _import_time()
    city = _StubCity(std_offset_hours=-5, cli_station="KNYC")
    summer = settlement_window(city, date(2024, 7, 15))
    # still computed from the STANDARD offset (-5), not the DST civil offset (-4):
    assert summer.start_utc.hour == 5
    assert summer.end_utc - summer.start_utc == timedelta(days=1)


def test_fields_populated():
    _SettlementWindow, settlement_window = _import_time()
    city = _StubCity(std_offset_hours=-8, cli_station="KLAX")
    w = settlement_window(city, date(2025, 1, 15))
    assert w.local_date == date(2025, 1, 15)
    assert w.std_offset_hours == -8
    assert w.station == "KLAX"
