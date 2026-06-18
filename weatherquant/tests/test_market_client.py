"""Signed WS + REST client reconnect discipline (PAP-01) — GREEN (05-02).

On reconnect the client must BOTH re-subscribe to ``orderbook_delta`` AND re-snapshot the
book (a reconnected stream must not reuse a stale local book — the seq baseline is gone,
T-05-09). The ``-k reconnect`` selector matches the VALIDATION map command ``pytest
tests/test_market_client.py -k reconnect``. Exercised with a MOCK WS + a MOCK HTTP client
(injectable seams) — no live network, no live creds.
"""

from __future__ import annotations

import pytest

from weatherquant.market import client as ws_client
from weatherquant.market.client import (
    REST_HOST_DEMO,
    REST_HOST_PROD,
    WS_URL_DEMO,
    WS_URL_PROD,
    fetch_snapshot,
    run_feed,
)

pytestmark = pytest.mark.asyncio


# --- Mocks (the injectable-seam idiom, no live network) ------------------------------------


class _MockResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _MockHttp:
    """Records every signed REST GET and returns a fixed ``orderbook_fp`` snapshot."""

    def __init__(self, payload):
        self._payload = payload
        self.calls = []  # (url, params, headers)

    async def get(self, url, *, params=None, headers=None):
        self.calls.append((url, params, headers))
        return _MockResponse(self._payload)


class _MockWS:
    """One WS connection: records sends, yields ``messages`` then raises ConnectionClosed."""

    def __init__(self, messages, *, close_after=True):
        self._messages = list(messages)
        self._close_after = close_after
        self.sent = []

    async def send(self, data):
        self.sent.append(data)

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for msg in self._messages:
            yield msg
        if self._close_after:
            import websockets

            raise websockets.ConnectionClosed(None, None)


class _MockConnector:
    """A ``websockets.connect``-shaped async iterator yielding scripted connections in order."""

    def __init__(self, connections):
        self._connections = list(connections)
        self.connect_calls = []  # (url, additional_headers)

    def __call__(self, url, *, additional_headers=None):
        self.connect_calls.append((url, additional_headers))
        return self

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for conn in self._connections:
            yield conn


def _signer(method, path):
    """A deterministic offline signer stub (no key material, no I/O)."""
    return {"KALSHI-ACCESS-KEY": "k", "KALSHI-ACCESS-SIGNATURE": "s", "KALSHI-ACCESS-TIMESTAMP": "1"}


_FP_PAYLOAD = {
    "orderbook_fp": {
        "seq": 100,
        "yes_dollars": [["0.47", "120"], ["0.46", "200"]],
        "no_dollars": [["0.49", "80"]],
    }
}

_TICKER = "KXHIGHNY-26JUN18-T72"


# --- Tests ---------------------------------------------------------------------------------


async def test_reconnect_resubscribes_and_resnapshots():
    """A reconnect re-subscribes to the feed AND re-snapshots (book not reused stale)."""
    http = _MockHttp(_FP_PAYLOAD)
    # Two scripted connections: the first drops after a delta, the second is the reconnect.
    conn1 = _MockWS(
        [{"type": "orderbook_delta", "seq": 101, "side": "yes", "price": 47, "delta": -20}]
    )
    conn2 = _MockWS([], close_after=False)
    connector = _MockConnector([conn1, conn2])

    seen = []
    await run_feed(
        [_TICKER],
        _signer,
        http=http,
        on_book=lambda t, b: seen.append((t, b.seq)),
        ws_connect=connector,
        max_reconnects=2,
    )

    # Subscribe command re-sent on BOTH connections.
    assert len(conn1.sent) == 1
    assert len(conn2.sent) == 1
    assert "subscribe" in conn1.sent[0]
    assert "subscribe" in conn2.sent[0]
    # fetch_snapshot re-invoked on reconnect: one REST GET per connection (2 total).
    assert len(http.calls) == 2


async def test_reconnect_uses_fresh_seq_baseline():
    """After reconnect the book's seq baseline comes from the fresh snapshot, not the old one."""
    http = _MockHttp(_FP_PAYLOAD)
    # First connection advances seq to 101 via a delta, then drops.
    conn1 = _MockWS(
        [{"type": "orderbook_delta", "seq": 101, "side": "yes", "price": 47, "delta": -20}]
    )
    conn2 = _MockWS([], close_after=False)
    connector = _MockConnector([conn1, conn2])

    captured = []
    await run_feed(
        [_TICKER],
        _signer,
        http=http,
        on_book=lambda t, b: captured.append(b.seq),
        ws_connect=connector,
        max_reconnects=2,
    )

    # After conn1's delta the book was at 101; the reconnect re-snapshots back to the REST seq
    # baseline (100) rather than carrying 101 forward — the stale book is discarded (T-05-09).
    # (The on_book callback fired once, at seq 101 on conn1.)
    assert captured == [101]


async def test_seq_gap_inside_loop_triggers_resnapshot():
    """A SeqGap during delta consumption discards + REST-resyncs the book (D-02)."""
    http = _MockHttp(_FP_PAYLOAD)
    # snapshot seq 100 -> delta 101 ok -> delta 104 GAP -> resnapshot, then no more.
    conn1 = _MockWS(
        [
            {"type": "orderbook_delta", "seq": 101, "side": "yes", "price": 47, "delta": -20},
            {"type": "orderbook_delta", "seq": 104, "side": "no", "price": 49, "delta": -30},
        ],
        close_after=False,
    )
    connector = _MockConnector([conn1])

    await run_feed([_TICKER], _signer, http=http, ws_connect=connector, max_reconnects=1)

    # Two REST GETs: the connect re-snapshot + the in-loop gap re-snapshot.
    assert len(http.calls) == 2


async def test_fetch_snapshot_signs_query_stripped_path_and_converts_to_cents():
    """REST snapshot signs the path WITHOUT the query and converts dollars -> cents."""
    signed_paths = []

    def recording_signer(method, path):
        signed_paths.append(path)
        return {"KALSHI-ACCESS-KEY": "k"}

    http = _MockHttp(_FP_PAYLOAD)
    snap = await fetch_snapshot(http, recording_signer, _TICKER, depth=10)

    # Signed path carries NO query string (Pitfall 6 / T-05-06).
    assert signed_paths == [f"/trade-api/v2/markets/{_TICKER}/orderbook"]
    assert "?" not in signed_paths[0]
    # depth went on the request params, not the signed path.
    assert http.calls[0][1] == {"depth": 10}
    # Dollar strings -> integer cents: 0.47 -> 47, 0.49 -> 49.
    assert snap["type"] == "orderbook_snapshot"
    assert snap["yes"] == [[47, 120], [46, 200]]
    assert snap["no"] == [[49, 80]]
    assert snap["seq"] == 100


async def test_hosts_are_fixed_constants():
    """The WS/REST hosts are fixed prod/demo constants (no host from untrusted input, T-05-11)."""
    assert WS_URL_PROD.startswith("wss://")
    assert WS_URL_DEMO.startswith("wss://")
    assert REST_HOST_PROD.startswith("https://")
    assert REST_HOST_DEMO.startswith("https://")
    # The module never builds a host from input — only these constants exist.
    assert ws_client._resolve_hosts(demo=False) == (WS_URL_PROD, REST_HOST_PROD)
    assert ws_client._resolve_hosts(demo=True) == (WS_URL_DEMO, REST_HOST_DEMO)
