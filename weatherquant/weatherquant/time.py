"""Fixed-offset LST settlement window — the single civil-time -> UTC primitive (TIME-01).

The ONE place a city's local settlement day becomes a UTC window; obs labeling and market
settlement both import :func:`settlement_window` from here.

THE DELIBERATE INVERSION (RESEARCH Pitfall 1 / D-01/D-02): a *fixed integer* standard offset
is MORE correct here than a DST-aware ``ZoneInfo`` conversion. Kalshi settles on the NWS Daily
Climate Report — a midnight-to-midnight Local Standard Time day, ignoring DST year-round; the
v3 bug used DST-aware math and shifted the day an hour during DST. So the window is computed
purely from ``city.std_offset_hours`` and this module imports neither ``zoneinfo`` nor
``timezonefinder`` (enforced by ``tests/test_no_runtime_dst.py``). The ``timedelta(days=1)``
span is exactly 24h precisely because there is no DST. During civil DST the window appears
shifted one civil-clock hour (TIME-02) — a consequence of the standard offset, not a branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from weatherquant.registry import City


@dataclass(frozen=True)
class SettlementWindow:
    """The UTC window for a city's local-standard-time settlement day (D-03).

    The window is half-open ``[start_utc, end_utc)`` — ``end_utc`` is EXCLUSIVE
    and equals the next day's ``start_utc``, so boundary observations are never
    double-counted.

    Attributes:
        local_date: The local (civil) settlement date the window represents.
        start_utc: tz-aware UTC instant of 00:00 LST on ``local_date``.
        end_utc: tz-aware UTC instant of 00:00 LST on ``local_date`` + 1 day,
            EXCLUSIVE (half-open). Always exactly 24h after ``start_utc``.
        std_offset_hours: The fixed standard offset used (e.g. -5 for ET).
        station: The CLI settlement station code (e.g. "KNYC").
    """

    local_date: date
    start_utc: datetime
    end_utc: datetime
    std_offset_hours: int
    station: str


def settlement_window(city: City, day: date) -> SettlementWindow:
    """Return the fixed-offset half-open UTC settlement window for ``day``.

    Pure arithmetic on ``city.std_offset_hours`` — no DST-aware conversion.
    Local-standard midnight expressed in UTC is ``local_midnight - offset``
    (the sign is ``- off``; US offsets are negative — RESEARCH Pitfall 3), so
    ``start_utc.hour == (-std_offset_hours) % 24``.
    """
    # std_offset_hours is whole-hour by contract — every in-scope Kalshi city is a
    # whole-hour standard offset. A half-hour zone (e.g. India +5:30, Newfoundland
    # -3:30) would need a minutes-based field; routed through this int-hours path it
    # would SILENTLY shift the window 30m — the same wrong-settlement-day failure
    # class Phase 1 exists to prevent. Switch registry.City to minutes before adding
    # any such city; do not coerce a fractional offset into this int.
    off = timedelta(hours=city.std_offset_hours)
    start_utc = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) - off
    end_utc = start_utc + timedelta(days=1)  # half-open [start, end); always 24h (no DST)
    return SettlementWindow(
        day, start_utc, end_utc, city.std_offset_hours, city.cli_station
    )
