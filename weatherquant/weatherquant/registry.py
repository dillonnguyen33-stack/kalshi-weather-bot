"""City / station registry — the 7 in-scope Kalshi daily-high cities.

This is a typed Python module (a frozen ``City`` dataclass + a ``CITIES`` dict)
rather than JSON/TOML (D-05): typed, importable, unit-testable, and free of any
runtime file I/O for a tiny static dataset. It is structured for trivial
extension to more cities later (D-06 — the registry is just more dict entries).

NO RUNTIME DST TOOLING. This module deliberately imports neither ``zoneinfo``
nor ``timezonefinder``. Each city stores a *fixed integer* ``std_offset_hours``
(standard-time offset from UTC, no DST) — that fixed int is the source of truth
for the settlement window (D-01). Offset *derivation and verification* against
``ZoneInfo`` on a January (standard-time) date lives ONLY in the tests
(``tests/test_registry.py``, D-02); re-introducing tz cleverness here would
recreate the v3 settlement bug and is caught by ``tests/test_no_runtime_dst.py``.

Stations are the VERIFIED Kalshi CLI settlement stations, read from Kalshi's own
published contract-term PDFs (01-RESEARCH § Verified Kalshi Settlement Stations),
NOT copied from v3. In particular Austin settles on Austin-Bergstrom (KAUS, NOT
Camp Mabry KATT) and Los Angeles on LAX Airport (KLAX, NOT Downtown/USC KCQT) —
do not "fix" those to the disproven values.

Coordinates and elevations are the CLI *station's* own lat/lon/elevation (Phase 2
extracts the GRIB point at the station, not the city centroid). Elevation is
REQUIRED (D-07) and stored in METERS (SI; RESEARCH lists both m and ft).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    """A single Kalshi settlement city. Field order/names per D-05.

    Attributes:
        lat: CLI station latitude (decimal degrees).
        lon: CLI station longitude (decimal degrees).
        elevation: CLI station elevation in METERS (required, D-07).
        cli_station: NWS Daily Climate Report station code (e.g. "KNYC").
        iana_tz: IANA timezone name — used in TESTS ONLY for the D-02
            January cross-check; never read on the runtime settlement path.
        std_offset_hours: Fixed standard-time offset from UTC in hours
            (e.g. -5 for Eastern). The single source of truth for the
            settlement window (D-01) — never a DST-aware conversion.
    """

    lat: float
    lon: float
    elevation: float
    cli_station: str
    iana_tz: str
    std_offset_hours: int


# Exactly the 7 in-scope Kalshi cities (D-06), keyed by Kalshi city code.
# Stations / offsets verified against Kalshi contract terms (01-RESEARCH).
# Elevation in meters. LA is locked to KLAX (the live KXHIGHLAX airport series);
# the 01-04 human-verify checkpoint resolved the KLAX-vs-KCQT residual risk
# (RESEARCH A1 / Open Question 1 / Pitfall 2) — see the LAX entry note below.
CITIES: dict[str, City] = {
    "NYC": City(40.779, -73.969, 48.0, "KNYC", "America/New_York", -5),
    "CHI": City(41.786, -87.752, 189.0, "KMDW", "America/Chicago", -6),
    "AUS": City(30.183, -97.680, 165.0, "KAUS", "America/Chicago", -6),
    "MIA": City(25.791, -80.316, 3.0, "KMIA", "America/New_York", -5),
    # LA series confirmed 2026-06-15: operator approved KLAX / KXHIGHLAX (LAXHIGH
    # airport series). KEEP the registry default — no KCQT / LA_DT entry added.
    "LAX": City(33.938, -118.389, 39.0, "KLAX", "America/Los_Angeles", -8),
    "DEN": City(39.847, -104.656, 1656.0, "KDEN", "America/Denver", -7),
    "PHI": City(39.872, -75.241, 11.0, "KPHL", "America/New_York", -5),
}


def get_city(code: str) -> City:
    """Resolve a Kalshi city code to its ``City`` (ASVS V5 input validation).

    Raises a clear ``KeyError`` on an unknown code rather than failing silently
    downstream (RESEARCH § Security Domain V5: reject unknown city codes).
    """
    try:
        return CITIES[code]
    except KeyError as exc:
        raise KeyError(
            f"unknown city code {code!r}; valid codes: {sorted(CITIES)}"
        ) from exc
