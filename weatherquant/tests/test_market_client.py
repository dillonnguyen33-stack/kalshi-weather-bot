"""Signed WS + REST client reconnect discipline (PAP-01) — verified-schema rewrite (05-12).

Two halves are exercised here against the live-verified Kalshi V2 protocol (05-UAT.md ##
Gaps → verified_live_schema):

* ``run_feed`` — the WS connection lifecycle. The B1/D-02 redesign (05-12): run_feed NO
  LONGER REST-resnapshots on connect (the seq-less REST orderbook_fp would crash the feed
  on the FIRST connect via ``_resolve_seq``'s "no seq baseline" raise). Each book is anchored
  by the fresh WS ``orderbook_snapshot`` (seq=1) that arrives after the subscribe command.
  A control frame (``subscribed``) is skipped before book keying. A WS seq gap BREAKS to the
  reconnect iterator, which re-subscribes on the NEXT connection for a fresh WS snapshot — it
  never REST-resyncs for a seq the REST API does not return.
* ``fetch_snapshot`` — the SIGNED REST orderbook read for ``run_paper``'s one-shot use. REST
  has NO seq, so the DEFAULT seq-less ``_FP_PAYLOAD`` fails loud ("no seq baseline", MED-5);
  the parse-success tests opt into a local seq-bearing payload to reach the dollars→cents +
  self-stamp code that must NOT regress.

Exercised with a MOCK WS + a MOCK HTTP client (injectable seams) — no live network, no
live creds. The ``-k reconnect`` selector still matches the VALIDATION map command.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from weatherquant.market import client as ws_client
from weatherquant.market.client import (
    REST_HOST_DEMO,
    REST_HOST_PROD,
    WS_URL_DEMO,
    WS_URL_PROD,
    _msg_ticker,
    fetch_snapshot,
    run_feed,
)

pytestmark = pytest.mark.asyncio


# --- Mocks (the injectable-seam idiom, no live network) ------------------------------------


class _MockResponse:
    def __init__(self, payload, *, headers=None):
        self._payload = payload
        # httpx responses expose ``.headers``; default empty so the now()-fallback path runs.
        self.headers = dict(headers or {})

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _MockHttp:
    """Records every signed REST GET and returns a fixed ``orderbook_fp`` snapshot.

    Optionally carries a ``headers`` mapping (e.g. a ``Date`` header) returned on every
    response so a test can exercise the server-observed-instant stamp.
    """

    def __init__(self, payload, *, headers=None):
        self._payload = payload
        self._headers = headers
        self.calls = []  # (url, params, headers)

    async def get(self, url, *, params=None, headers=None):
        self.calls.append((url, params, headers))
        return _MockResponse(self._payload, headers=self._headers)


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


_TICKER = "KXHIGHNY-26JUN18-T72"

# The VERIFIED live REST orderbook payload is SEQ-LESS — the orderbook_fp carries
# yes_dollars/no_dollars and NO "seq" key (05-UAT.md verified_live_schema). This is the
# DEFAULT for fetch_snapshot tests (run_paper's one-shot read); run_feed never consumes it
# (run_feed no longer REST-resnapshots, B1). On this default fetch_snapshot fails loud.
_FP_PAYLOAD = {
    "orderbook_fp": {
        "yes_dollars": [["0.47", "120"], ["0.46", "200"]],
        "no_dollars": [["0.49", "80"]],
    }
}

# A LOCAL seq-bearing REST variant for the few fetch_snapshot parse-SUCCESS tests that must
# reach the dollars→cents + self-stamp code (the verified-success path historically anchored
# on a REST seq). The seq-less default is the verified live shape; this opts a success-path
# test into a payload that parses without the "no seq baseline" raise.
_FP_PAYLOAD_WITH_SEQ = {
    "orderbook_fp": {
        "seq": 100,
        "yes_dollars": [["0.47", "120"], ["0.46", "200"]],
        "no_dollars": [["0.49", "80"]],
    }
}


# --- Enveloped WS message builders (verified live schema) ----------------------------------


def _ws_snapshot(seq=1, *, ticker=_TICKER, ts=None):
    """A VERIFIED enveloped WS orderbook_snapshot (book data under ``msg``, dollar strings)."""
    msg = {
        "market_ticker": ticker,
        "yes_dollars_fp": [["0.47", "120.00"], ["0.46", "200.00"]],
        "no_dollars_fp": [["0.49", "80.00"]],
    }
    if ts is not None:
        msg["ts"] = ts
    return {"type": "orderbook_snapshot", "sid": 1, "seq": seq, "msg": msg}


def _ws_delta(seq, *, side="yes", price="0.47", delta="-20.00", ticker=_TICKER, ts=None):
    """A VERIFIED enveloped WS orderbook_delta (single (side, price) change, dollar strings)."""
    msg = {
        "market_ticker": ticker,
        "side": side,
        "price_dollars": price,
        "delta_fp": delta,
    }
    if ts is not None:
        msg["ts"] = ts
    return {"type": "orderbook_delta", "sid": 1, "seq": seq, "msg": msg}


_CONTROL_FRAME = {"type": "subscribed", "id": 1, "msg": {"channel": "orderbook_delta", "sid": 1}}


# --- run_feed: B1 (no connect-time REST resnapshot) + control frame + WS-seq anchor --------


async def test_connect_does_not_rest_resnapshot_for_seq():
    """B1: run_feed runs to completion WITHOUT a connect-time REST GET (no seq-less crash).

    The mock WS yields a WS orderbook_snapshot (seq 1) + a contiguous delta (seq 2); the mock
    HTTP returns the SEQ-LESS orderbook_fp. The feed must NOT call fetch_snapshot for a seq
    baseline on connect (the seq-less payload would raise "no seq baseline" and kill the feed,
    the exact PAP-01 symptom) — it anchors on the WS snapshot instead.
    """
    http = _MockHttp(_FP_PAYLOAD)
    conn1 = _MockWS([_ws_snapshot(1), _ws_delta(2)], close_after=False)
    connector = _MockConnector([conn1])

    # Runs to completion without raising on the seq-less REST payload.
    await run_feed([_TICKER], _signer, http=http, ws_connect=connector, max_reconnects=1)

    # No connect-time REST GET — the seq baseline came from the WS snapshot, never REST (B1).
    assert http.calls == []


async def test_subscribed_control_frame_is_skipped():
    """A leading ``subscribed`` control frame is consumed without keying a book (multi-ticker).

    Driven with a MULTI-ticker feed so the missing ticker key on the control frame cannot be
    absorbed by the single-ticker shortcut; the following WS snapshot+delta apply normally.
    """
    http = _MockHttp(_FP_PAYLOAD)
    conn1 = _MockWS([_CONTROL_FRAME, _ws_snapshot(1), _ws_delta(2)], close_after=False)
    connector = _MockConnector([conn1])

    seen = []
    # Two tickers → the control frame's absent ticker key cannot fall through the shortcut.
    await run_feed(
        [_TICKER, "KX-OTHER"],
        _signer,
        http=http,
        on_book=lambda t, b: seen.append((t, b.seq)),
        ws_connect=connector,
        max_reconnects=1,
    )

    # The control frame keyed no book; the snapshot+delta surfaced the real ticker at seq 1, 2.
    assert (_TICKER, 1) in seen
    assert (_TICKER, 2) in seen
    assert http.calls == []


async def test_ws_snapshot_is_seq_anchor():
    """The WS orderbook_snapshot anchors the seq; contiguous deltas advance it (no REST)."""
    http = _MockHttp(_FP_PAYLOAD)
    conn1 = _MockWS([_ws_snapshot(1), _ws_delta(2), _ws_delta(3, side="no", price="0.49")],
                    close_after=False)
    connector = _MockConnector([conn1])

    captured = []
    await run_feed(
        [_TICKER],
        _signer,
        http=http,
        on_book=lambda t, b: captured.append(b.seq),
        ws_connect=connector,
        max_reconnects=1,
    )

    # Book ends at seq 3, anchored on the WS snapshot; the seq baseline never came from REST.
    assert captured[-1] == 3
    assert http.calls == []


async def test_seq_gap_resubscribes_for_fresh_ws_snapshot():
    """A WS seq gap breaks to the reconnect iterator, which re-subscribes for a fresh snapshot.

    conn1 yields WS snapshot(seq 1) → delta(seq 2) → delta(seq 5 GAP); conn2 is the reconnect
    with a fresh WS snapshot(seq 1). The deterministic break-to-reconnect mechanism (W2): the
    subscribe command is re-sent on conn2 AND fetch_snapshot is NOT called for the gap.
    """
    http = _MockHttp(_FP_PAYLOAD)
    conn1 = _MockWS([_ws_snapshot(1), _ws_delta(2), _ws_delta(5, side="no", price="0.49")])
    conn2 = _MockWS([_ws_snapshot(1)], close_after=False)
    connector = _MockConnector([conn1, conn2])

    await run_feed([_TICKER], _signer, http=http, ws_connect=connector, max_reconnects=2)

    # The gap broke conn1's loop; conn2 re-subscribed for a fresh WS snapshot (D-02 redesign).
    assert len(conn2.sent) == 1
    assert "subscribe" in conn2.sent[0]
    # fetch_snapshot was NOT called for the gap — REST is never the seq anchor.
    assert http.calls == []


async def test_reconnect_resubscribes():
    """A reconnect re-subscribes on the new connection (W3 — WS-seq, no REST resnapshot)."""
    http = _MockHttp(_FP_PAYLOAD)
    conn1 = _MockWS([_ws_snapshot(1), _ws_delta(2)])  # drops after a delta
    conn2 = _MockWS([_ws_snapshot(1)], close_after=False)  # the reconnect
    connector = _MockConnector([conn1, conn2])

    await run_feed([_TICKER], _signer, http=http, ws_connect=connector, max_reconnects=2)

    # Subscribe command re-sent on BOTH connections; never a REST GET (WS-seq anchor).
    assert len(conn1.sent) == 1
    assert len(conn2.sent) == 1
    assert "subscribe" in conn1.sent[0]
    assert "subscribe" in conn2.sent[0]
    assert http.calls == []


async def test_reconnect_uses_fresh_ws_seq_baseline():
    """After reconnect the seq baseline comes from the fresh WS snapshot, not a carried seq.

    conn1: snapshot(1) → delta(2) then drops; conn2: snapshot(1). The captured on_book seqs
    reflect the WS anchor (connect-snapshot@1, delta@2, reconnect-snapshot@1) — NOT a REST seq.
    The trailing 1 is the point: the reconnect re-anchors on the fresh WS snapshot (seq 1)
    rather than carrying conn1's stale 2 forward (T-05-09).
    """
    http = _MockHttp(_FP_PAYLOAD)
    conn1 = _MockWS([_ws_snapshot(1), _ws_delta(2)])
    conn2 = _MockWS([_ws_snapshot(1)], close_after=False)
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

    assert captured == [1, 2, 1]
    assert http.calls == []


async def test_event_time_surfaced_to_on_book():
    """PAP-03 (B2): on_book observes the book's real WS event time after a delta (never now()).

    A WS snapshot then a delta carrying ``msg.ts`` is applied; the on_book callback sees
    ``book.event_time`` == that tz-aware UTC instant. CARRY+SURFACE only — NOT DB persistence.
    """
    http = _MockHttp(_FP_PAYLOAD)
    ts = "2026-06-18T19:55:00.000000Z"
    conn1 = _MockWS([_ws_snapshot(1), _ws_delta(2, ts=ts)], close_after=False)
    connector = _MockConnector([conn1])

    seen_times = []
    await run_feed(
        [_TICKER],
        _signer,
        http=http,
        on_book=lambda t, b: seen_times.append(b.event_time),
        ws_connect=connector,
        max_reconnects=1,
    )

    expected = datetime(2026, 6, 18, 19, 55, 0, tzinfo=timezone.utc)
    # The real WS event time reached the sink after the delta (never back-dated to now()).
    assert expected in seen_times


# --- fetch_snapshot: verified REST behavior for run_paper's one-shot read (MUST NOT regress) -


async def test_fetch_snapshot_signs_query_stripped_path_and_converts_to_cents():
    """REST snapshot signs the path WITHOUT the query and converts dollars -> cents.

    Uses a LOCAL seq-bearing payload (the verified default is seq-less → fail loud); this
    test exercises the dollars→cents + self-stamp success path that must not regress.
    """
    signed_paths = []

    def recording_signer(method, path):
        signed_paths.append(path)
        return {"KALSHI-ACCESS-KEY": "k"}

    http = _MockHttp(_FP_PAYLOAD_WITH_SEQ)
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
    # CRIT-1: the snapshot self-stamps the observed instant — a tz-aware UTC datetime under
    # event_time AND its ISO string under snapshot_for (no Date header here → now() fallback).
    assert isinstance(snap["event_time"], datetime)
    assert snap["event_time"].tzinfo is not None
    assert snap["snapshot_for"] == snap["event_time"].isoformat()


async def test_fetch_snapshot_stamps_observed_instant_from_date_header():
    """CRIT-1: a server ``Date`` header IS the observed book instant (D-08, parsed to UTC)."""
    http = _MockHttp(_FP_PAYLOAD_WITH_SEQ, headers={"Date": "Wed, 18 Jun 2026 19:55:00 GMT"})
    snap = await fetch_snapshot(http, _signer, _TICKER)
    assert snap["event_time"] == datetime(2026, 6, 18, 19, 55, 0, tzinfo=timezone.utc)
    assert snap["snapshot_for"] == snap["event_time"].isoformat()


async def test_fetch_snapshot_missing_orderbook_fp_fails_loud():
    """HIGH-2: a renamed/absent ``orderbook_fp`` raises a descriptive ValueError, not KeyError."""
    http = _MockHttp({"orderbook": {"seq": 100, "yes_dollars": [], "no_dollars": []}})
    with pytest.raises(ValueError, match="orderbook_fp"):
        await fetch_snapshot(http, _signer, _TICKER)


async def test_fetch_snapshot_non_mapping_orderbook_fp_fails_loud():
    """HIGH-2: a non-Mapping ``orderbook_fp`` (list/None) raises a descriptive ValueError."""
    http = _MockHttp({"orderbook_fp": ["not", "a", "mapping"]})
    with pytest.raises(ValueError, match="orderbook_fp"):
        await fetch_snapshot(http, _signer, _TICKER)


async def test_fetch_snapshot_missing_seq_fails_loud():
    """MED-5: the VERIFIED seq-less REST orderbook_fp fails loud — never fabricate seq=0.

    This is the DEFAULT verified live shape: REST carries no seq, so run_paper's one-shot read
    fails loud rather than anchoring on a fabricated seq (the WS snapshot is the only seq anchor).
    """
    http = _MockHttp(_FP_PAYLOAD)  # the seq-less verified default
    with pytest.raises(ValueError, match="no seq baseline"):
        await fetch_snapshot(http, _signer, _TICKER)


async def test_fetch_snapshot_malformed_level_fails_loud():
    """HIGH-2: a malformed level (not a 2-element sequence) raises a descriptive ValueError."""
    http = _MockHttp(
        {"orderbook_fp": {"seq": 100, "yes_dollars": [["0.47"]], "no_dollars": []}}
    )
    with pytest.raises(ValueError, match="level"):
        await fetch_snapshot(http, _signer, _TICKER)


# --- _msg_ticker + host constants (unchanged behavior) -------------------------------------


async def test_msg_ticker_non_str_fails_loud():
    """TS-1: a present non-str ticker (an int market_id) raises rather than mis-keying.

    Driven with a MULTI-ticker feed so the single-ticker shortcut cannot short-circuit the
    guard: the untrusted ``market_id`` crosses the WS/REST trust boundary and would key
    ``books[ticker]`` — a non-str must fail loud, never silently mis-key/KeyError another
    ticker's book.
    """
    msg = {"type": "orderbook_delta", "market_id": 123, "seq": 5}
    with pytest.raises(ValueError, match="is not a str"):
        _msg_ticker(msg, ["KX-A", "KX-B"])


async def test_msg_ticker_str_passes_through():
    """A well-formed str ticker still resolves unchanged (happy path preserved)."""
    msg = {"type": "orderbook_delta", "market_id": "KX-A", "seq": 5}
    assert _msg_ticker(msg, ["KX-A", "KX-B"]) == "KX-A"


def test_hosts_are_fixed_constants():
    """The WS/REST hosts are fixed prod/demo constants (no host from untrusted input, T-05-11)."""
    assert WS_URL_PROD.startswith("wss://")
    assert WS_URL_DEMO.startswith("wss://")
    assert REST_HOST_PROD.startswith("https://")
    assert REST_HOST_DEMO.startswith("https://")
    # The module never builds a host from input — only these constants exist.
    assert ws_client._resolve_hosts(demo=False) == (WS_URL_PROD, REST_HOST_PROD)
    assert ws_client._resolve_hosts(demo=True) == (WS_URL_DEMO, REST_HOST_DEMO)
