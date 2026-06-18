"""Pessimistic maker queue (PAP-02, D-06) — 05-03 GREEN.

The maker queue is modeled PESSIMISTICALLY: cancels-ahead in the queue do NOT advance our
position (only a genuine trade-through does), and a maker order fills ONLY when a trade occurs
AND it crosses our resting level. Against ``scripted_book`` (size-ahead 100 at yes 50¢), a
cancel of 40 ahead leaves us still behind 100; only a trade that consumes the 100 size-ahead
AND crosses 50¢ fills us. The maker model is the conservative shadow (taker is the credited
Gate-1 path, D-05). ``fills.py`` is PURE.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from weatherquant.market import fills
from weatherquant.market.fills import BookEvent

WS_T1 = datetime(2026, 6, 18, 19, 55, tzinfo=timezone.utc)
WS_T2 = datetime(2026, 6, 18, 19, 56, tzinfo=timezone.utc)


def test_cancels_ahead_do_not_advance_queue(scripted_book):
    """A cancel ahead does NOT advance our queue position (pessimistic, D-06/Pitfall 3)."""
    size_ahead = scripted_book["size_ahead"][("yes", 50)]  # 100 resting ahead of us
    # A 40-contract cancel ahead, then a 30-contract trade that crosses: we are still behind
    # the original 100 (cancels did not advance us), so the trade consumes size-ahead, not us.
    events = [
        BookEvent(kind="cancel", size=40, crosses_our_level=False, event_time=WS_T1),
        BookEvent(kind="trade", size=30, crosses_our_level=True, event_time=WS_T2),
    ]
    fill = fills.maker_queue_fill(size_ahead=size_ahead, our_size=50, events=events)
    assert fill is None  # nothing reached us — the cancel never advanced our position


def test_maker_fill_only_on_trade_and_cross(scripted_book):
    """A maker order fills only when a trade consumes size-ahead AND crosses our level."""
    size_ahead = scripted_book["size_ahead"][("yes", 50)]  # 100
    # A 130-contract trade-through at/through our level: consumes the 100 ahead, the crossing
    # 30 fills our 50-contract order (partial), stamped with the trade's WS event time.
    events = [
        BookEvent(kind="trade", size=130, crosses_our_level=True, event_time=WS_T2),
    ]
    fill = fills.maker_queue_fill(size_ahead=size_ahead, our_size=50, events=events)
    assert fill is not None
    assert fill.is_maker is True
    assert fill.count == 30  # only the crossing remainder after size-ahead is consumed
    assert fill.partial is True
    assert fill.shortfall == 20
    assert fill.event_time == WS_T2  # real WS event time, never now()


def test_trade_not_crossing_does_not_fill():
    """A trade that consumes size-ahead but does NOT cross our level credits nothing."""
    events = [
        BookEvent(kind="trade", size=150, crosses_our_level=False, event_time=WS_T2),
    ]
    fill = fills.maker_queue_fill(size_ahead=100, our_size=50, events=events)
    assert fill is None


def test_no_trade_market_credits_zero_maker_fills():
    """A market with NO trades (only cancels) credits zero maker fills (never over-credit)."""
    events = [
        BookEvent(kind="cancel", size=60, crosses_our_level=False, event_time=WS_T1),
        BookEvent(kind="cancel", size=40, crosses_our_level=False, event_time=WS_T2),
    ]
    fill = fills.maker_queue_fill(size_ahead=100, our_size=50, events=events)
    assert fill is None
    # An empty event stream likewise credits nothing.
    assert fills.maker_queue_fill(size_ahead=100, our_size=50, events=[]) is None


def test_maker_fully_fills_when_crossing_exceeds_our_size():
    """A crossing trade larger than our size fills us fully (count == our_size, not partial)."""
    events = [
        BookEvent(kind="trade", size=300, crosses_our_level=True, event_time=WS_T2),
    ]
    fill = fills.maker_queue_fill(size_ahead=100, our_size=50, events=events)
    assert fill is not None
    assert fill.count == 50
    assert fill.partial is False
    assert fill.shortfall == 0


def test_maker_queue_fill_fail_loud_inputs():
    """Non-positive our_size or negative size_ahead fails loud (ValueError)."""
    with pytest.raises(ValueError, match="our_size must be positive"):
        fills.maker_queue_fill(size_ahead=100, our_size=0, events=[])
    with pytest.raises(ValueError, match="size_ahead must be non-negative"):
        fills.maker_queue_fill(size_ahead=-1, our_size=50, events=[])
