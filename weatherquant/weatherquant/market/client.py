"""Signed Kalshi WS client + REST one-shot orderbook read (PAP-01).

The live half of the orderbook feed:

* :func:`fetch_snapshot` — a SIGNED REST ``GET .../orderbook`` for ``run_paper``'s one-shot use;
  signs the path WITHOUT the ``?depth`` query (Pitfall 6 / T-05-06), parses ``orderbook_fp`` via
  the shared :func:`weatherquant.market.book.parse_dollar_fp_side` (W1). The live REST orderbook
  carries NO ``seq``, so :func:`_resolve_seq` fails loud (MED-5): REST is NOT a seq anchor.
* :func:`run_feed` — a MANUAL reconnect loop that RE-SIGNS the RSA-PSS handshake on every
  (re)connection (CR-01 — the ``async for ws in connect(...)`` idiom replays a stale
  ``KALSHI-ACCESS-TIMESTAMP`` Kalshi rejects). Each (re)connection re-sends subscribe; the fresh
  WS ``orderbook_snapshot`` (seq=1) ANCHORS each book (B1 — no connect-time REST resnapshot). A
  WS seq gap (:class:`SeqGap` / :class:`CorrectnessError`) BREAKS the delta loop so the next
  connection re-subscribes for a FRESH WS snapshot — never a REST resync (D-02 redesign).

SSRF discipline (T-05-11): WS/REST hosts are FIXED ``wss://`` / ``https://`` constants (ASVS
V9), never built from untrusted input. The signer, WS connector, and HTTP client are INJECTABLE
so tests drive a mock WS with no live network.
"""

from __future__ import annotations

import asyncio
import email.utils
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, UTC
from typing import Any

import httpx
import websockets

from weatherquant.ingest.errors import CorrectnessError
from weatherquant.market.book import (
    CONTROL_FRAME_TYPES,
    OrderBook,
    SeqGap,
    _TICKER_KEYS,
    apply,
    parse_dollar_fp_side,
)
from weatherquant.time import coerce_utc

logger = logging.getLogger(__name__)

# --- Fixed prod/demo hosts (SSRF guard T-05-11; NEVER built from untrusted input) ----------
WS_URL_PROD = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_URL_DEMO = "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"
REST_HOST_PROD = "https://external-api.kalshi.com"
REST_HOST_DEMO = "https://demo-api.kalshi.co"

# The WS handshake is signed over GET + this fixed path (no query — Pitfall 6).
WS_PATH = "/trade-api/ws/v2"
_WS_METHOD = "GET"

# Capped exponential backoff for run_feed's MANUAL reconnect loop (CR-01 can't use the
# websockets iterator's built-in backoff); applied only on unbounded (production) runs.
_RECONNECT_BACKOFF_BASE_SECONDS = 1.0
_RECONNECT_BACKOFF_MAX_SECONDS = 30.0

# REST orderbook path template; {ticker} is path-segment data, the host is a fixed const.
_REST_ORDERBOOK_PATH = "/trade-api/v2/markets/{ticker}/orderbook"
_REST_METHOD = "GET"

# Type aliases for the injectable seams.
SignerFn = Callable[[str, str], Mapping[str, str]]
OnBookFn = Callable[[str, OrderBook], Any]


def _resolve_hosts(demo: bool) -> tuple[str, str]:
    """Return ``(ws_url, rest_host)`` for the fixed prod or demo environment (SSRF guard)."""
    if demo:
        return WS_URL_DEMO, REST_HOST_DEMO
    return WS_URL_PROD, REST_HOST_PROD


def _observed_instant(response: Any) -> datetime:
    """Capture the server-observed book instant for a snapshot (CRIT-1, D-08).

    The REST observation time IS the book instant under D-08, captured at the fetch site so
    nothing downstream back-dates it. Prefers the server ``Date`` header (RFC-1123); falls back
    to ``datetime.now(timezone.utc)`` — the single sanctioned ``now()`` here.
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
            # A naive RFC-1123 date is GMT/UTC.
            return coerce_utc(parsed)
    return datetime.now(UTC)


def _require_fp(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fail-loud accessor for ``orderbook_fp`` — a missing or non-Mapping value raises a
    descriptive ValueError naming the field (ASVS V5, HIGH-2)."""
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
    """Resolve the snapshot's seq baseline (``fp.seq`` then ``payload.seq``), fail-loud if both
    absent (MED-5, D-02; a fabricated default would trip or mask SeqGap)."""
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

    The path is signed WITHOUT the ``?depth`` query (Pitfall 6 / T-05-06). Returns the
    ``orderbook_snapshot`` message shape :func:`weatherquant.market.book.apply` consumes, with
    integer-cent levels.

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
    headers = dict(signer(_REST_METHOD, path))  # query-stripped path (Pitfall 6)
    url = f"{rest_host}{path}"
    params = {"depth": depth} if depth else None
    # A transport error or malformed payload is logged [market error] and re-raised FAIL-LOUD
    # — run_paper needs a clear money-path failure, never a fabricated book (HIGH-2 / T-05-27).
    try:
        response = await http.get(url, params=params, headers=headers)
        response.raise_for_status()
        # Capture the observed book instant AT the fetch site (D-08).
        observed = _observed_instant(response)
        payload = response.json()
        fp = _require_fp(payload)
        return {
            "type": "orderbook_snapshot",
            "seq": _resolve_seq(fp, payload),
            "ticker": ticker,
            "yes": parse_dollar_fp_side(fp.get("yes_dollars")),
            "no": parse_dollar_fp_side(fp.get("no_dollars")),
            # The observed REST instant IS the book instant under D-08 (CRIT-1).
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
    """Surface a book state to the downstream sink, awaiting an awaitable result (CR-02).

    The ONE home of the ``on_book(ticker, book) + await-if-awaitable`` contract. A raising sink
    is caught and logged here so a sink hiccup does NOT kill the feed (WR-04).
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

    A MANUAL reconnect loop opens one connection at a time and RE-SIGNS the RSA-PSS handshake on
    EVERY (re)connection (CR-01). Each (re)connection re-sends the subscribe command; the fresh
    WS ``orderbook_snapshot`` (seq=1) ANCHORS each book (B1 — no connect-time REST resnapshot).

    Inside the delta loop:

    * a control frame (:data:`~weatherquant.market.book.CONTROL_FRAME_TYPES`) is skipped BEFORE
      ticker keying;
    * a snapshot/delta is routed through :func:`~weatherquant.market.book.apply` and surfaced to
      ``on_book``;
    * a :class:`SeqGap` / :class:`CorrectnessError` BREAKS the loop so the next connection
      re-subscribes for a FRESH WS snapshot — never a REST resync (D-02 redesign);
    * a malformed message fails loud; ``websockets.ConnectionClosed`` (and transient socket
      errors) log a fallback and reconnect.

    A capped exponential backoff applies between reconnects on unbounded (production) runs only,
    resetting after any connection that delivered data; bounded runs reconnect immediately.

    Args:
        tickers: the market tickers to subscribe to.
        signer: ``signer(method, path) -> headers``; RE-INVOKED once per (re)connection so each
            handshake carries a fresh timestamp (CR-01).
        http: injectable async HTTP client (kept for parity; run_feed issues no REST call).
        on_book: optional callback ``on_book(ticker, book)`` invoked after each applied
            snapshot/delta (sync or async); the book carries the WS event time on
            :attr:`OrderBook.event_time` after a delta (PAP-03).
        ws_connect: injectable WS connector returning a single async-context-manager connection;
            re-called per (re)connection. Defaults to ``websockets.connect``.
        demo: use the fixed demo hosts instead of prod (SSRF guard).
        max_reconnects: optional bound on (re)connections (primarily tests; bounded runs skip
            the backoff). ``None`` runs indefinitely.
    """
    ws_url, _ = _resolve_hosts(demo)
    connector = ws_connect or websockets.connect
    subscribe_cmd = _subscribe_command(tickers)
    # One book object per ticker, re-anchored in place so the `on_book` reference stays stable.
    books: dict[str, OrderBook] = {t: OrderBook(t) for t in tickers}

    connections = 0
    backoff = _RECONNECT_BACKOFF_BASE_SECONDS
    while max_reconnects is None or connections < max_reconnects:
        connections += 1
        # CR-01: re-sign the RSA-PSS handshake on EVERY (re)connection so the
        # KALSHI-ACCESS-TIMESTAMP is fresh (the auto-reconnect iterator replays a stale one).
        headers = dict(signer(_WS_METHOD, WS_PATH))
        delivered = False
        try:
            async with connector(ws_url, additional_headers=headers) as ws:
                # Re-subscribe; the fresh WS snapshot (seq=1) re-anchors each book (B1).
                await ws.send(subscribe_cmd)

                async for raw in ws:
                    try:
                        msg = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
                    except json.JSONDecodeError as exc:
                        # A single malformed frame is feed-level noise, not a connection fault:
                        # drop it and keep reading (like the control-frame skip below) — never let
                        # one bad frame kill the feed coroutine, and do NOT reconnect over it.
                        logger.warning("[ws error] dropping malformed WS frame: %s", exc)
                        continue
                    # A control frame is feed-level — skip it BEFORE keying a book (W1).
                    if isinstance(msg, Mapping) and msg.get("type") in CONTROL_FRAME_TYPES:
                        logger.debug("[ws] skipping control frame: %s", msg.get("type"))
                        continue
                    ticker = _msg_ticker(msg, tickers)
                    book = books[ticker]
                    try:
                        apply(book, msg)
                    except (SeqGap, CorrectnessError) as gap:
                        # The book is UNKNOWN across the gap — break so the NEXT connection
                        # re-subscribes for a FRESH WS snapshot, never REST-resyncing (D-02, W2).
                        logger.warning(
                            "[orderbook error] seq gap on %s (%s) — re-subscribing for a fresh WS "
                            "snapshot (D-02)",
                            ticker,
                            gap,
                        )
                        break
                    delivered = True
                    # Surface the applied book through _emit so a raising sink cannot kill the
                    # feed (WR-04); .event_time is the WS event time after a delta (PAP-03).
                    if on_book is not None:
                        await _emit(on_book, ticker, book)
        except websockets.ConnectionClosed as exc:
            # Expected transient: the loop reconnects (re-signing).
            logger.warning(
                "[ws error] kalshi WS connection closed (%s) — reconnecting + re-subscribing",
                exc,
            )
        # Unbounded runs back off between reconnects (reset after a connection that delivered
        # data); bounded runs reconnect immediately.
        if max_reconnects is None:
            backoff = _RECONNECT_BACKOFF_BASE_SECONDS if delivered else min(
                backoff * 2, _RECONNECT_BACKOFF_MAX_SECONDS
            )
            await asyncio.sleep(backoff)


def _msg_ticker(msg: Mapping[str, Any], tickers: Sequence[str]) -> str:
    """Resolve the ticker a message belongs to (tolerant key; single-ticker shortcut).

    The ticker keys ``books[ticker]`` — the trust boundary (TS-1); a present-but-non-str value
    FAILS LOUD. The WS envelope nests it under ``msg`` while REST carries it at the top level,
    so check the unwrapped body FIRST then the top level (A1, mirroring ``book._ticker_of``).
    """
    body = msg.get("msg") if isinstance(msg, Mapping) else None
    scopes = [body, msg] if isinstance(body, Mapping) else [msg]
    for scope in scopes:
        for key in _TICKER_KEYS:
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


__all__ = ["REST_HOST_DEMO", "REST_HOST_PROD", "WS_URL_DEMO", "WS_URL_PROD", "fetch_snapshot", "run_feed"]
