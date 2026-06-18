"""In-memory orderbook + seq integrity (PAP-01, D-02) — GREEN (05-02).

The book applies an ``orderbook_snapshot`` then contiguous ``orderbook_delta`` messages; a
gap in ``seq`` (next != last + 1) raises ``SeqGap`` (a ``CorrectnessError`` subclass) so the
client loop resnapshots rather than silently applying an out-of-order delta (D-02). The
``-k seq`` selector matches the VALIDATION map command ``pytest tests/test_market_book.py -k
seq``. Uses the ``orderbook_snapshot`` / ``orderbook_delta_stream`` /
``orderbook_delta_stream_with_gap`` conftest fixtures.
"""

from __future__ import annotations

import pytest

from weatherquant.ingest.errors import CorrectnessError
from weatherquant.market import reflect
from weatherquant.market.book import OrderBook, SeqGap, apply


def _levels_map(pairs):
    """Helper: collapse a book side's ``(price, count)`` pairs to a ``{price: count}`` dict."""
    return {price: count for price, count in pairs}


def test_book_applies_snapshot_then_contiguous_deltas(
    orderbook_snapshot, orderbook_delta_stream
):
    """Snapshot then in-order deltas keep the book consistent (no resnapshot)."""
    book = OrderBook()
    apply(book, orderbook_snapshot)

    # Snapshot reset: seq baseline + both bid sides materialized from the message.
    assert book.seq == 100
    assert book.ticker == "KXHIGHNY-26JUN18-T72"
    assert _levels_map(book.yes) == {47: 120, 46: 200, 45: 350}
    assert _levels_map(book.no) == {49: 80, 48: 150, 47: 260}

    for delta in orderbook_delta_stream:
        apply(book, delta)

    # seq advanced contiguously to the last applied delta.
    assert book.seq == 103
    # Deltas applied: yes47 120-20=100; no50 new level 60; yes48 new level 40.
    assert _levels_map(book.yes) == {47: 100, 48: 40, 46: 200, 45: 350}
    assert _levels_map(book.no) == {50: 60, 49: 80, 48: 150, 47: 260}


def test_book_level_drops_at_zero_total(orderbook_snapshot):
    """A delta that drives a level total to 0 (or below) DROPS the level (absence = absence)."""
    book = OrderBook()
    apply(book, orderbook_snapshot)
    # yes 47 has size 120 — remove exactly 120 → the level disappears entirely.
    apply(book, {"type": "orderbook_delta", "seq": 101, "side": "yes", "price": 47, "delta": -120})
    assert 47 not in _levels_map(book.yes)
    assert _levels_map(book.yes) == {46: 200, 45: 350}


def test_seq_gap_raises_seqgap(orderbook_snapshot, orderbook_delta_stream_with_gap):
    """A seq gap (101 then 104) raises ``SeqGap`` → the loop resnapshots (PAP-01, D-02)."""
    book = OrderBook()
    apply(book, orderbook_snapshot)
    # 101 is contiguous over the snapshot's seq 100 → applies fine.
    apply(book, orderbook_delta_stream_with_gap[0])
    assert book.seq == 101
    # 104 skips 102/103 → the local book is unknown; fail loud (never silently apply).
    with pytest.raises(SeqGap) as excinfo:
        apply(book, orderbook_delta_stream_with_gap[1])
    assert excinfo.value.expected == 102
    assert excinfo.value.got == 104


def test_seqgap_is_a_correctness_error():
    """``SeqGap`` subclasses ``CorrectnessError`` so the client loop catches it loudly."""
    assert issubclass(SeqGap, CorrectnessError)
    # IS-A ValueError too (mirrors the ingest errors' dual base / direct pytest.raises).
    assert issubclass(SeqGap, ValueError)


def test_delta_before_snapshot_fails_loud():
    """A delta on an uninitialized book raises — never coerce a fabricated level (V5, D-02)."""
    book = OrderBook()
    assert not book.initialized
    with pytest.raises(ValueError, match="before any orderbook_snapshot"):
        apply(book, {"type": "orderbook_delta", "seq": 101, "side": "yes", "price": 47, "delta": 5})


def test_unknown_message_type_fails_loud(orderbook_snapshot):
    """An unknown message ``type`` raises (fail loud, never coerce; T-05-10)."""
    book = OrderBook()
    apply(book, orderbook_snapshot)
    with pytest.raises(ValueError, match="unknown orderbook message type"):
        apply(book, {"type": "orderbook_fill", "seq": 101})


def test_missing_required_field_fails_loud(orderbook_snapshot):
    """A delta missing a required field raises rather than silently defaulting (V5)."""
    book = OrderBook()
    apply(book, orderbook_snapshot)
    with pytest.raises(ValueError, match="missing required field"):
        apply(book, {"type": "orderbook_delta", "seq": 101, "side": "yes", "price": 47})


def test_ticker_key_spelling_tolerated():
    """The ticker key is accepted as market_ticker / market_id / ticker (A1 defensive)."""
    for key in ("market_ticker", "market_id", "ticker"):
        book = OrderBook()
        apply(book, {"type": "orderbook_snapshot", "seq": 1, key: "KX-T", "yes": [], "no": []})
        assert book.ticker == "KX-T"


def test_book_feeds_reflect_seam(orderbook_snapshot):
    """The in-memory book works directly through the reflect seam (05-01 dict-or-attribute)."""
    book = OrderBook()
    apply(book, orderbook_snapshot)
    # best no bid 49¢/80 → yes ask 51¢/80 (best/cheapest first).
    yes_asks = reflect.yes_ask_levels(book)
    assert yes_asks[0] == (51, 80)
    # best yes bid 47¢/120 → no ask 53¢/120.
    no_asks = reflect.no_ask_levels(book)
    assert no_asks[0] == (53, 120)
