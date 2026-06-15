"""RED tests for the city/station registry (TIME-02 / D-02, criterion #3).

Imports ``weatherquant.registry`` (delivered by plan 01-02) — RED until then.

Behavior under test:
* ``CITIES`` has exactly 7 entries (D-06).
* Every ``City`` has non-null lat/lon/elevation/cli_station/iana_tz/std_offset_hours
  (criterion #3, D-07).
* D-02 January cross-check: each city's stored ``std_offset_hours`` equals the IANA
  zone's UTC offset on a known January (standard-time) date, derived via ``zoneinfo``.
  ``zoneinfo`` is allowed HERE (test/tooling tier) — never on the runtime path (D-02).
* The stations equal the VERIFIED Kalshi set {KNYC,KMDW,KAUS,KMIA,KLAX,KDEN,KPHL}.
  Explicitly asserts Austin == KAUS and LA == KLAX (guards against the disproven
  KATT / KCQT "fixes").
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo  # test/tooling tier only — NOT the runtime path (D-02)

import pytest

VERIFIED_STATIONS = {"KNYC", "KMDW", "KAUS", "KMIA", "KLAX", "KDEN", "KPHL"}

# Expected city codes — used to parametrize without importing the (not-yet-existing)
# registry at collection time. Each test imports CITIES/City lazily, so the RED signal
# is a per-test ModuleNotFoundError (weatherquant.registry lands in plan 01-02).
EXPECTED_CODES = ["NYC", "CHI", "AUS", "MIA", "LAX", "DEN", "PHI"]


def _import_registry():
    from weatherquant.registry import CITIES, City

    return CITIES, City


def test_exactly_seven_cities():
    CITIES, _City = _import_registry()
    assert len(CITIES) == 7, f"expected 7 cities, got {len(CITIES)}"


@pytest.mark.parametrize("code", EXPECTED_CODES)
def test_all_fields_populated(code):
    CITIES, City = _import_registry()
    city = CITIES[code]
    assert isinstance(city, City)
    assert city.lat is not None and isinstance(city.lat, float)
    assert city.lon is not None and isinstance(city.lon, float)
    assert city.elevation is not None  # criterion #3 / D-07 — elevation REQUIRED
    assert city.cli_station
    assert city.iana_tz
    assert isinstance(city.std_offset_hours, int)


@pytest.mark.parametrize("code", EXPECTED_CODES)
def test_std_offset_matches_january_zoneinfo(code):
    """D-02: stored fixed offset must equal the IANA zone's January (std-time) offset."""
    CITIES, _City = _import_registry()
    city = CITIES[code]
    jan = datetime(2025, 1, 15, 12, tzinfo=ZoneInfo(city.iana_tz))  # guaranteed std time
    derived = jan.utcoffset().total_seconds() / 3600
    assert derived == city.std_offset_hours, (
        f"{code}: stored {city.std_offset_hours} != zoneinfo January {derived}"
    )


def test_stations_are_the_verified_set():
    CITIES, _City = _import_registry()
    stations = {c.cli_station for c in CITIES.values()}
    assert stations == VERIFIED_STATIONS, f"unexpected stations: {stations}"


def test_austin_is_kaus_not_katt():
    """Guard: Austin settles on Austin-Bergstrom (KAUS), NOT Camp Mabry (KATT)."""
    CITIES, _City = _import_registry()
    assert CITIES["AUS"].cli_station == "KAUS"


def test_la_is_klax_not_kcqt():
    """Guard: the live KXHIGHLAX series settles on KLAX (Airport), NOT KCQT (Downtown)."""
    CITIES, _City = _import_registry()
    assert CITIES["LAX"].cli_station == "KLAX"
