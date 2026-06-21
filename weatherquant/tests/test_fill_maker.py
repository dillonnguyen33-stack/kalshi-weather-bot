"""Pessimistic maker queue (PAP-02, D-06) — 05-03 GREEN.

The maker queue is modeled PESSIMISTICALLY: cancels-ahead in the queue do NOT advance our
position (only a genuine trade-through does), and a maker order fills ONLY when a trade occurs
AND it crosses our resting level. Against ``scripted_book`` (size-ahead 100 at yes 50¢), a
cancel of 40 ahead leaves us still behind 100; only a trade that consumes the 100 size-ahead
AND crosses 50¢ fills us. The maker model is the conservative shadow (taker is the credited
Gate-1 path, D-05). ``fills.py`` is PURE.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from weatherquant.ingest.writer import WriteIntegrityError, insert_fill
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


def test_maker_fill_price_placeholder_is_nan_not_silent_zero():
    """A maker fill carries a NON-FINITE price placeholder, never a silent 0.0.

    The caller stamps the real resting price; an un-stamped 0.0 would corrupt CLV as
    ``closing_mid - 0`` (CORR-MED-4). NaN poisons any downstream arithmetic loudly instead.
    """
    events = [
        BookEvent(kind="trade", size=200, crosses_our_level=True, event_time=WS_T2),
    ]
    fill = fills.maker_queue_fill(size_ahead=100, our_size=50, events=events)
    assert fill is not None
    assert math.isnan(fill.avg_price_cents)


# maker_queue_fill returns avg_price_cents=NaN as the un-stamped placeholder (the queue model
# proves the COUNT; the caller supplies the real resting price, taker is the credited Gate-1
# path). The audited write path (writer.insert_fill, D-11) must therefore REFUSE a maker fill
# whose price is None/0/non-finite — persisting it would corrupt CLV as closing_mid - 0
# (CORR-MED-4). These precondition tests fire BEFORE any DB touch, so they need no live bind
# (the raise short-circuits _insert_row); bind=None is never reached.
@pytest.mark.parametrize("bad_price", [0, None, float("nan")])
def test_insert_fill_rejects_maker_zero_price(bad_price):
    """A maker fill with a fabricated 0c (or None) price fails loud (WriteIntegrityError)."""
    with pytest.raises(WriteIntegrityError, match="maker.*real resting price"):
        insert_fill(
            None,
            ticker="KXHIGHNY-26JUN18-T72",
            trade_id="t-maker-0",
            side="yes",
            price=bad_price,
            count=10,
            fee=2,
            is_maker=True,
            event_time=WS_T1,
            available_at=WS_T1,
        )


def test_insert_fill_maker_real_price_passes_guard():
    """A maker fill WITH a real non-zero resting price clears the guard (reaches the DB path).

    The guard is a pure precondition on is_maker+price; with a real price it must NOT raise
    WriteIntegrityError BEFORE the bind is used. We give a dummy bind that raises a DISTINCT
    error when touched, proving the maker-price guard let the call through to _insert_row.
    """

    class _SentinelBind:
        def __getattr__(self, _name):  # any DB access raises a distinguishable marker
            raise RuntimeError("reached-db")

    with pytest.raises(RuntimeError, match="reached-db"):
        insert_fill(
            _SentinelBind(),
            ticker="KXHIGHNY-26JUN18-T72",
            trade_id="t-maker-1",
            side="yes",
            price=50,
            count=10,
            fee=2,
            is_maker=True,
            event_time=WS_T1,
            available_at=WS_T1,
        )


def test_insert_fill_taker_zero_price_not_blocked_by_maker_guard():
    """A taker fill (is_maker=False) is unaffected by the maker-price guard (reaches the DB path)."""

    class _SentinelBind:
        def __getattr__(self, _name):
            raise RuntimeError("reached-db")

    with pytest.raises(RuntimeError, match="reached-db"):
        insert_fill(
            _SentinelBind(),
            ticker="KXHIGHNY-26JUN18-T72",
            trade_id="t-taker-0",
            side="yes",
            price=0,
            count=10,
            fee=2,
            is_maker=False,
            event_time=WS_T1,
            available_at=WS_T1,
        )
