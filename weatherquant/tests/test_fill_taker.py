"""Pessimistic taker sweep + partial fill (PAP-03, D-05/D-07) — 05-03 GREEN.

A taker order walks the reflected ask levels best-first, partial-fills when liquidity is
exhausted, and reports the SIZE-WEIGHTED average price (never an idealized single-price fill).
Against ``scripted_book`` yes bids (50¢×100, 49¢×60) a 130-contract taker sweep fills
100@50 + 30@49 → avg ``(100*50 + 30*49)/130 = 49.77¢``; a 200-contract order partial-fills
the 160 available and records the 40 shortfall. The ``-k sweep`` selector matches the
VALIDATION command ``pytest tests/test_fill_taker.py -k sweep``. ``fills.py`` is PURE.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from weatherquant.market import fills
from weatherquant.market.reflect import no_ask_levels

# A fixed WS event time stand-in (never ``now()``) stamped onto every produced fill.
WS_EVENT_TIME = datetime(2026, 6, 18, 19, 55, tzinfo=timezone.utc)


def test_taker_sweep_size_weighted_avg_price():
    """A 130-contract sweep over two levels credits the size-weighted avg of 49.77¢."""
    # Sweeping the YES-bid side directly as ask levels (already best-first in scripted_book).
    ask_levels = [(50, 100), (49, 60)]
    fill = fills.taker_sweep(ask_levels, 130, event_time=WS_EVENT_TIME)
    assert fill is not None
    assert fill.count == 130
    assert fill.partial is False
    assert fill.shortfall == 0
    assert fill.avg_price_cents == pytest.approx((100 * 50 + 30 * 49) / 130)
    assert fill.avg_price_cents == pytest.approx(49.769230769, abs=1e-6)


def test_taker_sweep_partial_fill_on_liquidity_exhaustion():
    """A 200-contract order partial-fills the 160 available, recording the 40 shortfall."""
    ask_levels = [(50, 100), (49, 60)]
    fill = fills.taker_sweep(ask_levels, 200, event_time=WS_EVENT_TIME)
    assert fill is not None
    assert fill.count == 160  # never fabricates the missing 40 contracts
    assert fill.partial is True
    assert fill.shortfall == 40
    assert fill.avg_price_cents == pytest.approx((100 * 50 + 60 * 49) / 160)


def test_taker_sweep_multilevel_size_weighted(scripted_book):
    """The sweep routes through the reflect seam: NO-ask reflects the YES bids (PAP-02)."""
    # no_ask = 100 - yes_bid: yes bids (50,100),(49,60) -> no asks (50,100),(51,60), best-first.
    ask_levels = no_ask_levels(scripted_book)
    assert ask_levels == [(50, 100), (51, 60)]
    fill = fills.taker_sweep(ask_levels, 130, event_time=WS_EVENT_TIME)
    assert fill is not None
    assert fill.count == 130
    assert fill.avg_price_cents == pytest.approx((100 * 50 + 30 * 51) / 130)


def test_taker_sweep_empty_book_returns_none():
    """An empty ask side fills nothing and returns None (absence = absence, no fill row)."""
    assert fills.taker_sweep([], 50, event_time=WS_EVENT_TIME) is None


def test_taker_sweep_exhausted_to_zero_returns_none():
    """A book of all-zero size credits zero fills and returns None, never a fabricated fill."""
    assert fills.taker_sweep([(50, 0), (49, 0)], 50, event_time=WS_EVENT_TIME) is None


def test_taker_sweep_rejects_nonpositive_want_count():
    """A non-positive want_count fails loud (ValueError)."""
    with pytest.raises(ValueError, match="want_count must be positive"):
        fills.taker_sweep([(50, 100)], 0, event_time=WS_EVENT_TIME)
    with pytest.raises(ValueError, match="want_count must be positive"):
        fills.taker_sweep([(50, 100)], -5, event_time=WS_EVENT_TIME)


def test_taker_sweep_rejects_negative_level_inputs():
    """A negative price or size in the book fails loud (ValueError)."""
    with pytest.raises(ValueError, match="non-negative"):
        fills.taker_sweep([(-1, 100)], 50, event_time=WS_EVENT_TIME)
    with pytest.raises(ValueError, match="non-negative"):
        fills.taker_sweep([(50, -100)], 50, event_time=WS_EVENT_TIME)
