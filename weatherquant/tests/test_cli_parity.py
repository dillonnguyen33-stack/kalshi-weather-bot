"""RED tests for CLI-report parity (D-04, TIME-02, criterion #2).

For each of the 7 cities and BOTH a winter day and a summer (civil-DST) day: bucket the
vendored hourly observations into ``settlement_window(city, day)``'s half-open
``[start_utc, end_utc)`` window, take the max, and assert it equals the fixture's CLI
"Maximum". Runs on vendored fixture data (Phase 1 proves the WINDOW math, not ingestion).

Imports ``weatherquant.time.settlement_window`` (plan 01-02) — RED until then. Uses the
``cli_fixture`` session fixture from conftest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pytest


@dataclass(frozen=True)
class _FixtureCity:
    std_offset_hours: int
    cli_station: str


def _parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


@pytest.mark.parametrize("season", ["winter", "summer"])
@pytest.mark.parametrize(
    "code", ["NYC", "CHI", "AUS", "MIA", "LAX", "DEN", "PHI"]
)
def test_window_max_matches_cli_maximum(cli_fixture, code, season):
    # RED: settlement_window lands in plan 01-02 (deferred import keeps collection green).
    from weatherquant.time import settlement_window

    payload = cli_fixture[code]
    city = _FixtureCity(
        std_offset_hours=payload["std_offset_hours"],
        cli_station=payload["station"],
    )
    day_data = payload["days"][season]
    day = date.fromisoformat(day_data["date"])

    w = settlement_window(city, day)

    in_window = [
        o["temp_f"]
        for o in day_data["obs"]
        if w.start_utc <= _parse_ts(o["ts_utc"]) < w.end_utc  # half-open [start, end)
    ]
    assert in_window, f"{code}/{season}: no obs fell inside the window"
    assert max(in_window) == day_data["cli_max"], (
        f"{code}/{season}: window max {max(in_window)} != CLI maximum "
        f"{day_data['cli_max']} (a sign error or inclusive end would pick up the "
        f"out-of-window trap reading)."
    )
