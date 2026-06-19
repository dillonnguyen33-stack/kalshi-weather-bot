"""Signed Kalshi WS client + REST one-shot orderbook read (PAP-01).

This is the live half of the orderbook feed:

* :func:`fetch_snapshot` — a SIGNED REST ``GET /trade-api/v2/markets/{ticker}/orderbook``
  that reads a ticker's book from scratch for ``run_paper``'s ONE-SHOT use. It signs the path
  WITHOUT the ``?depth`` query (Pitfall 6 / threat T-05-06) and parses the ``orderbook_fp``
  dollar-string pairs into the ONE internal CENT representation via the SINGLE shared parser
  :func:`weatherquant.market.book.parse_dollar_fp_side` (W1 — no second dollars→cents math
  lives here). The verified live REST orderbook carries NO ``seq`` field, so
  :func:`_resolve_seq` raises fail-loud ("no seq baseline", MED-5): REST is NOT a seq anchor.
* :func:`run_feed` — the connection lifecycle: a MANUAL reconnect loop opens one signed
  connection at a time and RE-SIGNS the RSA-PSS handshake on every (re)connection (CR-01 — the
  ``async for ws in connect(...)`` auto-reconnect idiom freezes the first handshake's headers
  and replays a stale ``KALSHI-ACCESS-TIMESTAMP`` that Kalshi rejects, so reconnect-and-anchor
  could never re-authenticate live). On EVERY (re)connection the loop body re-sends the
  subscribe command; the fresh WS ``orderbook_snapshot`` (seq=1) that arrives after subscribe is
  what ANCHORS each book (B1 — run_feed does NOT REST-resnapshot on connect; the seq-less REST
  payload would crash the feed on the first connect). A control frame
  (``subscribed``/``ok``/``error``/``unsubscribed``) is skipped before book keying. A WS seq gap
  (:class:`SeqGap` / :class:`~weatherquant.ingest.errors.CorrectnessError`) BREAKS the delta loop
  so the loop reconnects to the NEXT connection, which re-subscribes for a FRESH WS snapshot (a
  new per-subscription seq baseline) — never a REST resync for a seq the REST API does not
  return (D-02 redesign, live-verified).

SSRF discipline (threat T-05-11, mirrors ``ingest/sources/_client.py``): the WS and REST
hosts are FIXED prod/demo constants — never built from untrusted input — and are ``wss://`` /
``https://`` (TLS only, ASVS V9). The signed-handshake header builder, the WS connector, and
the HTTP client are all INJECTABLE so the unit tests drive a mock WS that drops once and
asserts the re-subscribe + WS-snapshot re-anchor happen on reconnect, with NO live network.
"""

from __future__ import annotations

import asyncio
import email.utils
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import httpx
import websockets

from weatherquant.ingest.errors import CorrectnessError
from weatherquant.market.book import (
    CONTROL_FRAME_TYPES,
    OrderBook,
    SeqGap,
    apply,
    parse_dollar_fp_side,
)

logger = logging.getLogger(__name__)

# --- Fixed prod/demo hosts (SSRF guard T-05-11; NEVER built from untrusted input) ----------
WS_URL_PROD = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_URL_DEMO = "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"
REST_HOST_PROD = "https://external-api.kalshi.com"
REST_HOST_DEMO = "https://demo-api.kalshi.co"

# The WS handshake is signed over GET + this fixed path (no query — Pitfall 6).
WS_PATH = "/trade-api/ws/v2"
_WS_METHOD = "GET"

# Reconnect pacing for run_feed's MANUAL reconnect loop. The loop re-signs the handshake on
# every (re)connection (CR-01), so it cannot delegate to the websockets auto-reconnect
# iterator's built-in backoff; this is a capped exponential backoff reproduced in its place,
# applied only on unbounded (production) runs — bounded test runs reconnect immediately.
_RECONNECT_BACKOFF_BASE_SECONDS = 1.0
_RECONNECT_BACKOFF_MAX_SECONDS = 30.0

# REST orderbook path template; the {ticker} is path-segment data, the host is a fixed const.
_REST_ORDERBOOK_PATH = "/trade-api/v2/markets/{ticker}/orderbook"
_REST_METHOD = "GET"

# Type aliases for the injectable seams (so tests can supply mocks with no live network).
SignerFn = Callable[[str, str], Mapping[str, str]]
OnBookFn = Callable[[str, OrderBook], Any]


def _resolve_hosts(demo: bool) -> tuple[str, str]:
    """Return ``(ws_url, rest_host)`` for the fixed prod or demo environment (SSRF guard)."""
    if demo:
        return WS_URL_DEMO, REST_HOST_DEMO
    return WS_URL_PROD, REST_HOST_PROD


def _parse_fp_side(raw: Any) -> list[list[int]]:
    """Parse one ``orderbook_fp`` side to cent levels — delegates to the ONE shared parser (W1).

    The dollar/fp side parse lives in exactly ONE home now:
    :func:`weatherquant.market.book.parse_dollar_fp_side` (exported by 05-11). This thin wrapper
    preserves the call sites in :func:`fetch_snapshot` while removing the duplicate dollars→cents
    conversion that previously lived here — no second ``round(p*100)`` math survives in this
    module (W1 grep gate). The price → integer cents and the count → ``round(float(...))``
    fail-loud semantics are unchanged (they are the shared parser's, asserted by book's tests).
    """
    return parse_dollar_fp_side(raw)


def _observed_instant(response: Any) -> datetime:
    """Capture the server-observed book instant for a snapshot (CRIT-1, D-08).

    WHY this is the legitimate live-now fence: under D-08 the real REST observation time IS the
    observed book instant — captured HERE at the I/O edge (the fetch site), mirroring
    ``ingest/available_at.py``'s LIVE branch, so nothing downstream ever back-dates it. Prefer
    the server ``Date`` header (RFC-1123) when present and parseable; fall back to
    ``datetime.now(timezone.utc)`` — the single sanctioned ``now()`` in this module — captured
    at the fetch site.
    """
    raw_date = None
    headers = getattr(response, "headers", None)
    if headers is not None:
        raw_date = headers.get("Date")
    if raw_date:
        try:
            parsed = email.utils.parsedate_to_datetime(raw_date)
        except (TypeError, ValueError):
            parsed = None
        if parsed is not None:
            # A naive RFC-1123 date is GMT/UTC; normalize tz-aware results to UTC.
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _require_fp(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fail-loud accessor for ``orderbook_fp`` (mirror book._msg_get; name the missing field).

    A renamed/absent ``orderbook_fp`` raises a descriptive ValueError naming the field (never a
    bare KeyError); a non-Mapping ``orderbook_fp`` (a list/None) also raises (ASVS V5, HIGH-2).
    """
    if "orderbook_fp" not in payload:
        raise ValueError(
            f"orderbook snapshot missing required field 'orderbook_fp': {payload!r}"
        )
    fp = payload["orderbook_fp"]
    if not isinstance(fp, Mapping):
        raise ValueError(
            f"orderbook_fp must be a mapping of seq/yes_dollars/no_dollars, got {fp!r}"
        )
    return fp


def _resolve_seq(fp: Mapping[str, Any], payload: Mapping[str, Any]) -> int:
    """Resolve the snapshot's seq baseline, fail-loud on absence (MED-5, D-02).

    A snapshot anchors delta integrity (D-02); a fabricated ``seq=0`` would spuriously trip
    SeqGap on the first real delta or mask a genuine gap. Read ``fp.seq`` then ``payload.seq``;
    if BOTH are absent raise rather than default (absence = absence).
    """
    seq = fp.get("seq")
    if seq is None:
        seq = payload.get("seq")
    if seq is None:
        raise ValueError(
            "orderbook snapshot carries no seq baseline — cannot anchor delta integrity (D-02)"
        )
    return int(seq)


async def fetch_snapshot(
    http: Any,
    signer: SignerFn,
    ticker: str,
    *,
    depth: int = 0,
    rest_host: str = REST_HOST_PROD,
) -> dict[str, Any]:
    """Fetch + sign a REST orderbook snapshot for ``ticker`` (the resync anchor, PAP-01).

    The path is signed WITHOUT the ``?depth`` query (Pitfall 6 / T-05-06); ``depth`` is sent
    only as a request param. The ``orderbook_fp`` dollar-string pairs are converted to the ONE
    internal CENT representation and returned in the ``orderbook_snapshot`` message shape the
    book's :func:`weatherquant.market.book.apply` consumes (``type``/``seq``/``yes``/``no``).

    Args:
        http: an injectable async HTTP client exposing ``await http.get(url, params=, headers=)``
            (an ``httpx.AsyncClient`` in production, a mock in tests).
        signer: the signing seam ``signer(method, path) -> headers`` (e.g.
            :meth:`weatherquant.market.auth.KalshiSigner.sign`).
        ticker: the market ticker (a path segment; the host stays a fixed const, T-05-11).
        depth: optional orderbook depth (0 = full book; not part of the signed path).
        rest_host: the fixed REST host constant (prod by default; demo for the checkpoint).

    Returns:
        A snapshot-shaped dict: ``{"type": "orderbook_snapshot", "seq", "ticker", "yes", "no",
        "event_time", "snapshot_for"}`` with integer-cent bid levels. ``event_time`` is the
        tz-aware UTC observed book instant (CRIT-1, D-08) and ``snapshot_for`` is its ISO string.

    Raises:
        ValueError: on a malformed/renamed payload (missing/non-Mapping ``orderbook_fp``, a
            malformed level, or a missing seq baseline) — fail loud, never a fabricated book.
        httpx.HTTPError: on a transport/HTTP error (logged ``[market error]`` and re-raised).
    """
    path = _REST_ORDERBOOK_PATH.format(ticker=ticker)
    headers = dict(signer(_REST_METHOD, path))  # sign the QUERY-STRIPPED path (Pitfall 6)
    url = f"{rest_host}{path}"
    params = {"depth": depth} if depth else None
    # The whole network call + untrusted-JSON parse is wrapped in the house-style try/except
    # (mirror book._msg_get / _coerce_levels discipline): a transport error or a malformed
    # payload is logged with a [market error] prefix and re-raised FAIL-LOUD — the run_paper
    # call site needs a clear money-path failure, never a fabricated book (HIGH-2 / T-05-27).
    try:
        response = await http.get(url, params=params, headers=headers)
        response.raise_for_status()
        # Capture the observed book instant AT the fetch site (D-08, the live-now fence here).
        observed = _observed_instant(response)
        payload = response.json()
        fp = _require_fp(payload)
        return {
            "type": "orderbook_snapshot",
            "seq": _resolve_seq(fp, payload),
            "ticker": ticker,
            "yes": _parse_fp_side(fp.get("yes_dollars")),
            "no": _parse_fp_side(fp.get("no_dollars")),
            # The server-observed instant: the real REST observation time IS the book instant
            # under D-08, stamped HERE so nothing downstream back-dates it (CRIT-1).
            "event_time": observed,
            "snapshot_for": observed.isoformat(),
        }
    except (httpx.HTTPError, KeyError, ValueError, TypeError) as exc:
        logger.warning("[market error] fetch_snapshot failed for %s: %s", ticker, exc)
        raise


def _subscribe_command(tickers: Sequence[str]) -> str:
    """Build the ``orderbook_delta`` subscribe command JSON for ``tickers``."""
    return json.dumps(
        {
            "id": 1,
            "cmd": "subscribe",
            "params": {"channels": ["orderbook_delta"], "market_tickers": list(tickers)},
        }
    )


async def _emit(on_book: OnBookFn, ticker: str, book: OrderBook) -> None:
    """Surface a book state to the downstream sink, awaiting a coroutine result (WR-04 wrapping).

    The ONE place the ``on_book(ticker, book) + await-if-awaitable`` contract lives, called from
    the per-message emit site (after each applied WS snapshot/delta) so the contract cannot
    drift again (CR-02). A raise from the downstream sink (e.g. a transient DB write error in the
    persist sink) is caught and logged here rather than allowed to tear down the whole live feed:
    a sink hiccup must NOT kill the orderbook feed; the caller continues/reconnects (WR-04).
    """
    try:
        result = on_book(ticker, book)
        if isinstance(result, Awaitable):
            await result
    except Exception as exc:  # noqa: BLE001 — the sink must not kill the feed (WR-04)
        logger.warning("[market error] on_book sink failed for %s: %s", ticker, exc)


async def run_feed(
    tickers: Sequence[str],
    signer: SignerFn,
    *,
    http: Any,
    on_book: OnBookFn | None = None,
    ws_connect: Callable[..., Any] | None = None,
    demo: bool = False,
    max_reconnects: int | None = None,
) -> None:
    """Run the signed WS orderbook feed with re-signed reconnects + re-subscribe + WS anchor.

    A MANUAL reconnect loop owns socket reconnection: it opens one connection at a time and
    RE-SIGNS the RSA-PSS handshake on EVERY (re)connection (CR-01 — the ``async for ws in
    connect(...)`` auto-reconnect idiom would freeze the first connection's ``additional_headers``
    and replay a stale ``KALSHI-ACCESS-TIMESTAMP``, which the live server rejects on its
    timestamp-skew window). On EVERY (re)connection the loop body re-sends the ``orderbook_delta``
    subscribe command; the fresh WS ``orderbook_snapshot`` (seq=1) that arrives right after the
    subscribe ANCHORS each book (B1 — run_feed does NOT REST-resnapshot on connect; the verified
    live REST orderbook_fp carries no seq and would crash the feed via the "no seq baseline"
    raise). The WS snapshot, not REST, is the per-subscription seq baseline.

    Inside the delta loop:

    * a control frame (``type`` in :data:`~weatherquant.market.book.CONTROL_FRAME_TYPES`) is
      skipped (``continue``) BEFORE ticker keying — it is feed-level, not ticker-scoped;
    * a WS ``orderbook_snapshot``/``orderbook_delta`` is routed through
      :func:`~weatherquant.market.book.apply` and the book is surfaced to ``on_book``;
    * a :class:`SeqGap` / :class:`~weatherquant.ingest.errors.CorrectnessError` BREAKS the delta
      loop so the loop reconnects to the NEXT connection, which re-subscribes for a FRESH WS
      snapshot (a new per-subscription seq baseline) — the unknown book is never carried forward
      and REST is never consulted for a seq it does not return (D-02 redesign);
    * a malformed message fails loud; only ``websockets.ConnectionClosed`` (and other transient
      socket errors) log a structured fallback and let the loop reconnect.

    Between (re)connections the loop applies a capped exponential backoff on unbounded
    (production) runs only — bounded runs (``max_reconnects`` set, primarily tests) reconnect
    immediately. The backoff resets after any connection that delivered book data.

    Args:
        tickers: the market tickers to subscribe to.
        signer: ``signer(method, path) -> headers`` for the signed WS handshake; RE-INVOKED once
            per (re)connection so each handshake carries a fresh timestamp (CR-01).
        http: injectable async HTTP client (kept for parity with :func:`fetch_snapshot`'s seam;
            run_feed itself issues no REST call — the WS snapshot is the only seq anchor).
        on_book: optional callback ``on_book(ticker, book)`` invoked after each applied WS
            snapshot/delta (the downstream fill/snapshot sink); may be sync or async. The book
            it receives carries the real WS event time on :attr:`OrderBook.event_time` after a
            delta (PAP-03 carry+surface; the WS→persistence sink is deferred — B2).
        ws_connect: injectable WS connector — ``ws_connect(url, additional_headers=...)`` returns
            a SINGLE connection usable as an async context manager (the ``websockets.connect``
            idiom); re-called per (re)connection so the handshake can be re-signed. A mock in
            tests. Defaults to ``websockets.connect``.
        demo: use the fixed demo hosts instead of prod (SSRF guard; both are constants).
        max_reconnects: optional bound on (re)connections — primarily for tests so a mock WS
            that always reconnects does not loop forever (and so bounded runs skip the reconnect
            backoff). ``None`` runs indefinitely with backoff between reconnects.
    """
    ws_url, _ = _resolve_hosts(demo)
    connector = ws_connect or websockets.connect
    subscribe_cmd = _subscribe_command(tickers)
    # One book object per ticker; the WS snapshot re-anchors it in place, never replaced
    # wholesale, so the downstream `on_book` reference stays stable across reconnects.
    books: dict[str, OrderBook] = {t: OrderBook(t) for t in tickers}

    connections = 0
    backoff = _RECONNECT_BACKOFF_BASE_SECONDS
    while max_reconnects is None or connections < max_reconnects:
        connections += 1
        # CR-01: re-sign the RSA-PSS handshake on EVERY (re)connection so the
        # KALSHI-ACCESS-TIMESTAMP is fresh. The websockets auto-reconnect iterator would freeze
        # the first handshake's headers and replay a stale signature the live server rejects, so
        # the whole reconnect/re-subscribe/re-anchor path could never re-authenticate. A manual
        # loop re-invokes the signer here, opening one freshly-signed connection at a time.
        headers = dict(signer(_WS_METHOD, WS_PATH))
        delivered = False
        try:
            async with connector(ws_url, additional_headers=headers) as ws:
                # (Re)connect: re-subscribe; the fresh WS snapshot (seq=1) re-anchors each book.
                # No connect-time REST resnapshot (B1) — REST has no seq, the WS snapshot anchors.
                await ws.send(subscribe_cmd)

                async for raw in ws:
                    msg = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
                    # A control frame ('subscribed'/'ok'/'error'/'unsubscribed') is feed-level and
                    # carries no ticker key — skip it BEFORE _msg_ticker keys a book (the verified
                    # first frame after subscribe; one home for the set in book.py — W1).
                    if isinstance(msg, Mapping) and msg.get("type") in CONTROL_FRAME_TYPES:
                        logger.debug("[ws] skipping control frame: %s", msg.get("type"))
                        continue
                    ticker = _msg_ticker(msg, tickers)
                    book = books[ticker]
                    try:
                        apply(book, msg)
                    except (SeqGap, CorrectnessError) as gap:
                        # The local book is UNKNOWN across the gap — break so the loop reconnects
                        # and the NEXT connection re-subscribes for a FRESH WS snapshot (new seq
                        # baseline), never REST-resyncing for a seq REST does not return (D-02
                        # redesign, W2). Do NOT re-subscribe on this same socket; do NOT fetch.
                        logger.warning(
                            "[orderbook error] seq gap on %s (%s) — re-subscribing for a fresh WS "
                            "snapshot (D-02)",
                            ticker,
                            gap,
                        )
                        break
                    delivered = True
                    # Surface the freshly applied book (snapshot or delta) to the sink. The book's
                    # .event_time is the real WS event time after a delta (PAP-03 carry+surface,
                    # B2); the WS→persistence sink is deferred (not wired here). Routed through the
                    # one _emit helper so a raising sink cannot kill the feed (WR-04).
                    if on_book is not None:
                        await _emit(on_book, ticker, book)
        except websockets.ConnectionClosed as exc:
            # EXPECTED transient: the loop reconnects (re-signing); structured fallback.
            logger.warning(
                "[ws error] kalshi WS connection closed (%s) — reconnecting + re-subscribing",
                exc,
            )
        # Reconnect pacing: unbounded (production) runs back off between reconnects so a
        # persistently failing endpoint is not hammered, resetting after any connection that
        # delivered book data; bounded runs (tests) reconnect immediately for determinism.
        if max_reconnects is None:
            backoff = _RECONNECT_BACKOFF_BASE_SECONDS if delivered else min(
                backoff * 2, _RECONNECT_BACKOFF_MAX_SECONDS
            )
            await asyncio.sleep(backoff)


def _msg_ticker(msg: Mapping[str, Any], tickers: Sequence[str]) -> str:
    """Resolve the ticker a message belongs to (tolerant key; single-ticker shortcut).

    The ticker value is UNTRUSTED WS/REST JSON that immediately keys ``books[ticker]`` in
    ``run_feed`` — the trust boundary (TS-1). A present ticker key whose value is not a str
    (e.g. an int ``market_id``) FAILS LOUD here rather than silently mis-keying or
    KeyErroring another ticker's book; the ``-> str`` annotation is now actually enforced.

    The verified live WS envelope nests the ticker under the ``msg`` body
    (``msg.market_ticker``), while the REST one-shot snapshot shape from :func:`fetch_snapshot`
    carries it at the top level (``ticker``). So check the unwrapped ``msg`` body FIRST, then the
    top level — mirroring :func:`weatherquant.market.book._ticker_of`'s envelope read so both
    sides agree on where the ticker lives (one tolerated set of spellings, A1).
    """
    body = msg.get("msg") if isinstance(msg, Mapping) else None
    scopes = [body, msg] if isinstance(body, Mapping) else [msg]
    for scope in scopes:
        for key in ("market_ticker", "market_id", "ticker"):
            if key in scope:
                value = scope[key]
                if not isinstance(value, str):
                    raise ValueError(
                        f"orderbook ticker under {key!r} is not a str: {value!r}"
                    )
                return value
    if len(tickers) == 1:
        return tickers[0]
    raise ValueError(
        f"orderbook message carries no ticker key and feed has multiple tickers: {msg!r}"
    )


__all__ = ["run_feed", "fetch_snapshot", "WS_URL_PROD", "WS_URL_DEMO", "REST_HOST_PROD", "REST_HOST_DEMO"]
