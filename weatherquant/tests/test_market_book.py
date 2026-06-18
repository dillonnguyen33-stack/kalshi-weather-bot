"""RED stub — in-memory orderbook + seq integrity (PAP-01, 05-02 fills this GREEN).

The book applies an ``orderbook_snapshot`` then contiguous ``orderbook_delta`` messages;
a gap in ``seq`` (next != last + 1) raises ``SeqGap`` (a ``CorrectnessError`` subclass) so
the client loop resnapshots rather than silently applying an out-of-order delta (D-02). The
``-k seq`` selector matches the VALIDATION map command ``pytest tests/test_market_book.py -k
seq``.

Wave-0 RED stub: ``importorskip`` the not-yet-existing ``weatherquant.market.book``. Uses the
``orderbook_snapshot`` / ``orderbook_delta_stream`` / ``orderbook_delta_stream_with_gap``
conftest fixtures so 05-02 flips it GREEN against fixed contracts.
"""

from __future__ import annotations

import pytest

book = pytest.importorskip("weatherquant.market.book")


@pytest.mark.xfail(reason="RED — 05-02 implements the book", strict=False)
def test_book_applies_snapshot_then_contiguous_deltas(
    orderbook_snapshot, orderbook_delta_stream
):
    """Snapshot then in-order deltas keep the book consistent (no resnapshot)."""
    raise NotImplementedError("05-02: apply snapshot, then each delta in seq order")


@pytest.mark.xfail(reason="RED — 05-02 implements the book", strict=False)
def test_seq_gap_raises_seqgap(orderbook_snapshot, orderbook_delta_stream_with_gap):
    """A seq gap (101 then 104) raises ``SeqGap`` → the loop resnapshots (PAP-01, D-02)."""
    raise NotImplementedError("05-02: detect next_seq != last_seq + 1, raise SeqGap")


@pytest.mark.xfail(reason="RED — 05-02 implements the book", strict=False)
def test_seqgap_is_a_correctness_error():
    """``SeqGap`` subclasses ``CorrectnessError`` so the client loop catches it loudly."""
    raise NotImplementedError("05-02: SeqGap(CorrectnessError)")
