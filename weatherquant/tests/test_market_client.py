"""RED stub — signed WS + REST client reconnect discipline (PAP-01, 05-02 fills this GREEN).

On reconnect the client must BOTH re-subscribe to ``orderbook_delta`` AND re-snapshot the
book (a reconnected stream must not reuse a stale local book — the seq baseline is gone).
The ``-k reconnect`` selector matches the VALIDATION map command ``pytest
tests/test_market_client.py -k reconnect``. Exercised with a mock WS (injectable client, the
``ingest/sources/_client.py`` MockTransport idiom) — no live network.

Wave-0 RED stub: ``importorskip`` the not-yet-existing ``weatherquant.market.client``.
"""

from __future__ import annotations

import pytest

client = pytest.importorskip("weatherquant.market.client")


@pytest.mark.xfail(reason="RED — 05-02 implements the WS client", strict=False)
def test_reconnect_resubscribes_and_resnapshots():
    """A reconnect re-subscribes to the feed AND re-snapshots (book not reused stale)."""
    raise NotImplementedError("05-02: async for ws in connect(): subscribe + REST snapshot")


@pytest.mark.xfail(reason="RED — 05-02 implements the WS client", strict=False)
def test_reconnect_uses_fresh_seq_baseline():
    """After reconnect the book's seq baseline comes from the fresh snapshot, not the old one."""
    raise NotImplementedError("05-02: the resnapshot resets the book seq baseline")
