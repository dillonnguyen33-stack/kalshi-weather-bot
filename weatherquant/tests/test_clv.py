"""Closing-line value (PAP-04, 05-04 GREEN).

CLV measures a fill against the volume-weighted CLOSING mid over the final window before the
market settles. Sign: a yes BUY filled BELOW the closing mid is POSITIVE CLV (we bought
cheap), above is NEGATIVE. The closing mid is golden over ``closing_window_snapshots``
(volume-weighted = 51.4¢). The ``-k window`` selector matches the VALIDATION command ``pytest
tests/test_clv.py -k window``: the closing window MUST be anchored on
``time.settlement_window(city, day).end_utc`` (the half-open EXCLUSIVE end) — NEVER a
hand-rolled civil-time clock (D-10, the v3 founding bug). ``clv.py`` is a PURE module.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

clv = pytest.importorskip("weatherquant.market.clv")

from weatherquant.registry import get_city  # noqa: E402
from weatherquant.time import settlement_window  # noqa: E402


@dataclass(frozen=True)
class _Fill:
    """Minimal structural fill carrying the size-weighted avg price (cents)."""

    avg_price_cents: float


def test_volume_weighted_closing_mid_golden(closing_window_snapshots):
    """The closing mid is the volume-weighted mean (51.4¢) over the window snapshots."""
    assert clv.vol_weighted_mid(closing_window_snapshots) == pytest.approx(51.4)


def test_clv_sign_positive_when_fill_better_than_close(closing_window_snapshots):
    """A yes BUY at 48¢ vs a 51.4¢ closing mid yields POSITIVE CLV (we bought cheap)."""
    result = clv.clv_cents(_Fill(48.0), closing_window_snapshots, "buy")
    assert result == pytest.approx(51.4 - 48.0)
    assert result > 0.0


def test_clv_sign_negative_when_fill_worse_than_close(closing_window_snapshots):
    """A yes BUY at 55¢ vs a 51.4¢ closing mid yields NEGATIVE CLV (we overpaid)."""
    result = clv.clv_cents(_Fill(55.0), closing_window_snapshots, "buy")
    assert result == pytest.approx(51.4 - 55.0)
    assert result < 0.0


def test_clv_sign_flips_for_sell(closing_window_snapshots):
    """A SELL at 55¢ vs the 51.4¢ close is POSITIVE (sold dear); the sign flips from buy."""
    sell = clv.clv_cents(_Fill(55.0), closing_window_snapshots, "sell")
    buy = clv.clv_cents(_Fill(55.0), closing_window_snapshots, "buy")
    assert sell == pytest.approx(-buy)
    assert sell > 0.0


def test_empty_closing_window_fails_loud():
    """An empty closing window raises rather than fabricating a mid (absence = absence)."""
    with pytest.raises(ValueError):
        clv.vol_weighted_mid([])


def test_zero_volume_closing_window_fails_loud():
    """A zero-total-volume window raises rather than dividing by zero / fabricating a mid."""
    with pytest.raises(ValueError):
        clv.vol_weighted_mid([{"mid": 50.0, "volume": 0}])


def test_closing_window_anchored_on_settlement_window_end():
    """The closing window anchors on ``settlement_window(...).end_utc``, half-open EXCLUSIVE.

    A snapshot AT ``end_utc`` is EXCLUDED (half-open); one inside the final
    ``CLV_WINDOW_MINUTES`` is INCLUDED; one before the window start is EXCLUDED. This proves
    the LST clock anchor (D-10) — not a re-derived civil-time clock.
    """
    city = get_city("NYC")
    day = date(2026, 6, 18)
    win = settlement_window(city, day)
    end = win.end_utc
    window_minutes = clv.CLV_WINDOW_MINUTES

    inside = end - timedelta(minutes=window_minutes // 2)  # within the half-open window
    at_start = end - timedelta(minutes=window_minutes)  # the inclusive lower edge
    before = end - timedelta(minutes=window_minutes + 5)  # below the window
    at_end = end  # the EXCLUSIVE upper edge — must be dropped

    snaps = [
        {"event_time": before, "mid": 40.0, "volume": 10, "tag": "before"},
        {"event_time": at_start, "mid": 50.0, "volume": 10, "tag": "at_start"},
        {"event_time": inside, "mid": 52.0, "volume": 10, "tag": "inside"},
        {"event_time": at_end, "mid": 99.0, "volume": 10, "tag": "at_end"},
    ]

    selected = clv.closing_window_snapshots(snaps, city, day)
    tags = {s["tag"] for s in selected}
    assert tags == {"at_start", "inside"}  # half-open: at_start in, at_end out, before out


def test_closing_window_parses_iso_snapshot_for():
    """The window selection also works on the ``snapshot_for`` ISO string shape (fixtures)."""
    city = get_city("NYC")
    day = date(2026, 6, 18)
    win = settlement_window(city, day)
    inside = (win.end_utc - timedelta(minutes=5)).isoformat()
    after = win.end_utc.isoformat()
    snaps = [
        {"snapshot_for": inside, "mid": 52.0, "volume": 10, "tag": "inside"},
        {"snapshot_for": after, "mid": 99.0, "volume": 10, "tag": "after"},
    ]
    selected = clv.closing_window_snapshots(snaps, city, day)
    assert [s["tag"] for s in selected] == ["inside"]
