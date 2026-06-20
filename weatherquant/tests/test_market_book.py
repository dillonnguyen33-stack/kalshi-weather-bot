"""In-memory orderbook + seq integrity (PAP-01, D-02) — the VERIFIED enveloped contract (05-11).

The book applies a Kalshi V2 ``orderbook_snapshot`` then contiguous ``orderbook_delta`` messages
in the shape VERIFIED LIVE in 05-UAT.md (## Gaps → verified_live_schema): an ENVELOPE whose book
data is nested under ``msg`` as dollar/fixed-point STRING levels, with a PER-SUBSCRIPTION ``seq``
anchored on the WS snapshot (snapshot=1, first delta=2). A ``subscribed`` control frame is IGNORED
(not fail-loud); a gap in ``seq`` raises ``SeqGap`` (a ``CorrectnessError`` subclass); a delta
carries the real WS event time (``msg.ts``/``msg.ts_ms``) onto ``OrderBook.event_time`` (PAP-03).
The ``-k seq`` selector matches the VALIDATION map command ``pytest tests/test_market_book.py -k
seq``. Uses the ``orderbook_snapshot`` / ``orderbook_delta_stream`` /
``orderbook_delta_stream_with_gap`` / ``control_frame`` conftest fixtures.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from weatherquant.ingest.errors import CorrectnessError
from weatherquant.market import reflect
from weatherquant.market.book import OrderBook, SeqGap, _ticker_of, apply


def _levels_map(pairs):
    """Helper: collapse a book side's ``(price, count)`` pairs to a ``{price: count}`` dict."""
    return {price: count for price, count in pairs}


def test_book_applies_snapshot_then_contiguous_deltas(
    orderbook_snapshot, orderbook_delta_stream
):
    """Enveloped snapshot then in-order deltas keep the book consistent (no resnapshot)."""
    book = OrderBook()
    apply(book, orderbook_snapshot)

    # Snapshot reset: WS seq anchor (1) + both bid sides parsed from the dollar/fp strings.
    assert book.seq == 1
    assert book.ticker == "KXHIGHNY-26JUN18-T72"
    assert _levels_map(book.yes) == {47: 120, 46: 200, 45: 350}
    assert _levels_map(book.no) == {49: 80, 48: 150, 47: 260}

    for delta in orderbook_delta_stream:
        apply(book, delta)

    # seq advanced contiguously to the last applied delta (snapshot 1 → 2 → 3 → 4).
    assert book.seq == 4
    # Deltas applied: yes47 120-20=100; no50 new level 60; yes48 new level 40.
    assert _levels_map(book.yes) == {47: 100, 48: 40, 46: 200, 45: 350}
    assert _levels_map(book.no) == {50: 60, 49: 80, 48: 150, 47: 260}


def test_book_level_drops_at_zero_total(orderbook_snapshot):
    """A delta that drives a level total to 0 (or below) DROPS the level (absence = absence)."""
    book = OrderBook()
    apply(book, orderbook_snapshot)
    # yes 47 has size 120 — remove exactly 120 → the level disappears entirely.
    apply(
        book,
        {
            "type": "orderbook_delta", "sid": 1, "seq": 2,
            "msg": {"side": "yes", "price_dollars": "0.47", "delta_fp": "-120.00"},
        },
    )
    assert 47 not in _levels_map(book.yes)
    assert _levels_map(book.yes) == {46: 200, 45: 350}


def test_control_frame_ignored(orderbook_snapshot, control_frame):
    """A ``subscribed`` control frame is IGNORED — mutates nothing, never fail-loud (05-11)."""
    # An uninitialized book stays uninitialized after a control frame.
    fresh = OrderBook()
    apply(fresh, control_frame)
    assert not fresh.initialized
    assert fresh.seq is None
    assert fresh.yes == [] and fresh.no == []

    # An initialized book is unchanged by a control frame (seq + levels untouched).
    book = OrderBook()
    apply(book, orderbook_snapshot)
    before_seq, before_yes, before_no = book.seq, _levels_map(book.yes), _levels_map(book.no)
    apply(book, control_frame)
    assert book.seq == before_seq
    assert _levels_map(book.yes) == before_yes
    assert _levels_map(book.no) == before_no


def test_seq_gap_raises_seqgap(orderbook_snapshot, orderbook_delta_stream_with_gap):
    """A seq gap (2 then 5) raises ``SeqGap`` → book-level raise only (PAP-01, D-02, W2)."""
    book = OrderBook()
    apply(book, orderbook_snapshot)
    # seq 2 is contiguous over the snapshot's seq 1 → applies fine.
    apply(book, orderbook_delta_stream_with_gap[0])
    assert book.seq == 2
    # seq 5 skips 3/4 → the local book is unknown; fail loud (never silently apply).
    with pytest.raises(SeqGap) as excinfo:
        apply(book, orderbook_delta_stream_with_gap[1])
    assert excinfo.value.expected == 3
    assert excinfo.value.got == 5


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
        apply(
            book,
            {
                "type": "orderbook_delta", "sid": 1, "seq": 2,
                "msg": {"side": "yes", "price_dollars": "0.47", "delta_fp": "5.00"},
            },
        )


def test_unknown_message_type_fails_loud(orderbook_snapshot):
    """An unknown NON-control message ``type`` raises (fail loud, never coerce; T-05-10)."""
    book = OrderBook()
    apply(book, orderbook_snapshot)
    # ``orderbook_fill`` is neither a known data type NOR a recognized control frame.
    with pytest.raises(ValueError, match="unknown orderbook message type"):
        apply(book, {"type": "orderbook_fill", "sid": 1, "seq": 2, "msg": {}})


def test_missing_required_field_fails_loud(orderbook_snapshot):
    """A delta missing ``msg.price_dollars`` raises rather than silently defaulting (V5)."""
    book = OrderBook()
    apply(book, orderbook_snapshot)
    with pytest.raises(ValueError, match="missing required field"):
        apply(
            book,
            {
                "type": "orderbook_delta", "sid": 1, "seq": 2,
                "msg": {"side": "yes", "delta_fp": "5.00"},
            },
        )


def test_ticker_key_spelling_tolerated():
    """The ticker key (under ``msg``) is accepted as market_ticker / market_id / ticker (A1)."""
    for key in ("market_ticker", "market_id", "ticker"):
        book = OrderBook()
        apply(
            book,
            {
                "type": "orderbook_snapshot", "sid": 1, "seq": 1,
                "msg": {key: "KX-T", "yes_dollars_fp": [], "no_dollars_fp": []},
            },
        )
        assert book.ticker == "KX-T"


def test_ticker_of_non_str_fails_loud():
    """TS-1: a present non-str ticker key (in the unwrapped msg body) raises (no mis-key).

    ``_ticker_of`` is annotated ``-> str | None``; a present-but-non-str ticker value (an int
    ``market_id``) must fail loud so the declared type is actually enforced, never coerced. The
    ticker now lives under ``msg``, so ``_ticker_of`` reads from the unwrapped msg body.
    """
    with pytest.raises(ValueError, match="is not a str"):
        _ticker_of({"market_id": 123})


def test_ticker_of_absent_key_returns_none():
    """A1 tolerance preserved: an ABSENT ticker key still returns None (not a raise)."""
    assert _ticker_of({"yes_dollars_fp": []}) is None


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


def test_event_time_carried(orderbook_snapshot, orderbook_delta_stream):
    """A delta's ``msg.ts``/``msg.ts_ms`` is carried onto ``OrderBook.event_time`` (PAP-03)."""
    book = OrderBook()
    apply(book, orderbook_snapshot)
    # The first delta carries ts "2026-06-18T19:55:00.000000Z" → tz-aware UTC datetime.
    apply(book, orderbook_delta_stream[0])
    assert isinstance(book.event_time, datetime)
    assert book.event_time.tzinfo is not None
    assert book.event_time == datetime(2026, 6, 18, 19, 55, 0, tzinfo=timezone.utc)
