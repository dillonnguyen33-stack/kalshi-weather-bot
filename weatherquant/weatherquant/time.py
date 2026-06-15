"""Fixed-offset LST settlement window — the single civil-time -> UTC primitive.

This module is the ONE place the codebase converts a city's local settlement day
into a UTC window (TIME-01). Both observation labeling (Phase 2) and market
settlement (Phase 6) import ``settlement_window`` from here; no other module
re-derives civil time.

THE DELIBERATE INVERSION (RESEARCH Pitfall 1): a *fixed integer* standard offset
is MORE correct here than a DST-aware ``ZoneInfo`` conversion. Kalshi settles on
the NWS Daily Climate Report, whose climatological day is midnight-to-midnight
Local Standard Time, ignoring DST year-round. The v3 bug was computing the day
with DST-aware tz math, which shifted the day by an hour during DST. So this
module computes the window purely arithmetically from ``city.std_offset_hours``
and MUST NOT import ``zoneinfo`` or ``timezonefinder`` (D-01/D-02; enforced by
``tests/test_no_runtime_dst.py``). The fixed ``timedelta(days=1)`` span is
correct precisely BECAUSE there is no DST in this window — the standard-offset
day is always exactly 24h.

During civil DST the standard-offset window appears shifted one civil-clock hour
(TIME-02), but that is an observed consequence of using the standard offset
year-round, not a separate code branch.
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
    off = timedelta(hours=city.std_offset_hours)
    start_utc = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) - off
    end_utc = start_utc + timedelta(days=1)  # half-open [start, end); always 24h (no DST)
    return SettlementWindow(
        day, start_utc, end_utc, city.std_offset_hours, city.cli_station
    )
