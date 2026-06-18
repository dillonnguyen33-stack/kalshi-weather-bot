"""Signed Kalshi WS client + REST snapshot resync anchor (PAP-01).

This is the live half of the orderbook feed:

* :func:`fetch_snapshot` — a SIGNED REST ``GET /trade-api/v2/markets/{ticker}/orderbook``
  that rebuilds a ticker's book from scratch. It signs the path WITHOUT the ``?depth`` query
  (Pitfall 6 / threat T-05-06) and parses the ``orderbook_fp`` dollar-string pairs into the
  ONE internal CENT representation the WS feed and the book also use
  (:mod:`weatherquant.market.book`).
* :func:`run_feed` — the connection lifecycle: ``async for ws in connect(...)`` lets the
  ``websockets`` iterator own socket reconnection + backoff, but on EVERY (re)connection the
  loop body re-subscribes AND forces a REST re-snapshot of each ticker BEFORE resuming deltas
  (the iterator reconnects the socket, not the book — threat T-05-09). A :class:`SeqGap` /
  :class:`~weatherquant.ingest.errors.CorrectnessError` inside the delta loop discards the
  affected book and re-snapshots (D-02); only genuinely transient errors log a structured
  fallback and let the iterator reconnect.

SSRF discipline (threat T-05-11, mirrors ``ingest/sources/_client.py``): the WS and REST
hosts are FIXED prod/demo constants — never built from untrusted input — and are ``wss://`` /
``https://`` (TLS only, ASVS V9). The signed-handshake header builder, the WS connector, and
the HTTP client are all INJECTABLE so the unit tests drive a mock WS that drops once and
asserts the re-subscribe + re-snapshot happen on reconnect, with NO live network.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

import websockets

from weatherquant.ingest.errors import CorrectnessError
from weatherquant.market.book import OrderBook, SeqGap, apply

logger = logging.getLogger(__name__)

# --- Fixed prod/demo hosts (SSRF guard T-05-11; NEVER built from untrusted input) ----------
WS_URL_PROD = "wss://external-api-ws.kalshi.com/trade-api/ws/v2"
WS_URL_DEMO = "wss://external-api-ws.demo.kalshi.co/trade-api/ws/v2"
REST_HOST_PROD = "https://external-api.kalshi.com"
REST_HOST_DEMO = "https://demo-api.kalshi.co"

# The WS handshake is signed over GET + this fixed path (no query — Pitfall 6).
WS_PATH = "/trade-api/ws/v2"
_WS_METHOD = "GET"

# REST orderbook path template; the {ticker} is path-segment data, the host is a fixed const.
_REST_ORDERBOOK_PATH = "/trade-api/v2/markets/{ticker}/orderbook"
_REST_METHOD = "GET"

# Dollars → cents: Kalshi REST returns price/size as dollar strings; converge to integer cents.
_DOLLARS_TO_CENTS = 100

# Type aliases for the injectable seams (so tests can supply mocks with no live network).
SignerFn = Callable[[str, str], Mapping[str, str]]
OnBookFn = Callable[[str, OrderBook], Any]


def _resolve_hosts(demo: bool) -> tuple[str, str]:
    """Return ``(ws_url, rest_host)`` for the fixed prod or demo environment (SSRF guard)."""
    if demo:
        return WS_URL_DEMO, REST_HOST_DEMO
    return WS_URL_PROD, REST_HOST_PROD


def _cents(dollar_string: str | float | int) -> int:
    """Convert a dollar price/size string to integer cents (``round(p*100)``)."""
    return round(float(dollar_string) * _DOLLARS_TO_CENTS)


def _parse_fp_side(raw: Any) -> list[list[int]]:
    """Parse one ``orderbook_fp`` side (``[[price_dollars, count], ...]``) to cent levels.

    Only the PRICE is a dollar amount → converted to integer cents (``round(p*100)``); the
    COUNT is a raw contract/share count, taken as an integer as-is (it is NOT a dollar amount,
    so it must never be ×100 — that would inflate every resting size). Fails loud on a
    malformed level (never coerce a fabricated level; ASVS V5, T-05-10). A missing side
    (``None``) is an empty book side (absence = absence).
    """
    if raw is None:
        return []
    levels: list[list[int]] = []
    for level in raw:
        price_d, count = level  # fail loud if not a 2-tuple
        levels.append([_cents(price_d), round(float(count))])
    return levels


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
        A snapshot-shaped dict: ``{"type": "orderbook_snapshot", "seq", "ticker", "yes", "no"}``
        with integer-cent bid levels.
    """
    path = _REST_ORDERBOOK_PATH.format(ticker=ticker)
    headers = dict(signer(_REST_METHOD, path))  # sign the QUERY-STRIPPED path (Pitfall 6)
    url = f"{rest_host}{path}"
    params = {"depth": depth} if depth else None
    response = await http.get(url, params=params, headers=headers)
    response.raise_for_status()
    payload = response.json()
    fp = payload["orderbook_fp"]
    return {
        "type": "orderbook_snapshot",
        "seq": int(fp.get("seq", payload.get("seq", 0))),
        "ticker": ticker,
        "yes": _parse_fp_side(fp.get("yes_dollars")),
        "no": _parse_fp_side(fp.get("no_dollars")),
    }


def _subscribe_command(tickers: Sequence[str]) -> str:
    """Build the ``orderbook_delta`` subscribe command JSON for ``tickers``."""
    return json.dumps(
        {
            "id": 1,
            "cmd": "subscribe",
            "params": {"channels": ["orderbook_delta"], "market_tickers": list(tickers)},
        }
    )


async def _resnapshot_all(
    books: Mapping[str, OrderBook],
    http: Any,
    signer: SignerFn,
    tickers: Sequence[str],
    *,
    rest_host: str,
) -> None:
    """REST-resnapshot every ticker's book in place (the reconnect / seq-gap resync, D-02)."""
    for ticker in tickers:
        snapshot = await fetch_snapshot(http, signer, ticker, rest_host=rest_host)
        apply(books[ticker], snapshot)


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
    """Run the signed WS orderbook feed with auto-reconnect + re-subscribe + re-snapshot.

    The ``websockets`` iterator (``async for ws in connect(...)``) owns socket reconnection and
    backoff. On EVERY (re)connection the loop body:

    1. re-sends the ``orderbook_delta`` subscribe command, AND
    2. REST-re-snapshots every ticker's book (rebuilds the seq baseline from a fresh snapshot;
       the reconnected socket must NOT reuse the stale local book — threat T-05-09),

    BEFORE consuming any deltas. Inside the delta loop a :class:`SeqGap` /
    :class:`~weatherquant.ingest.errors.CorrectnessError` discards + re-snapshots the affected
    book (D-02); a malformed message fails loud; only ``websockets.ConnectionClosed`` (and
    other transient socket errors) log a structured fallback and let the iterator reconnect.

    Args:
        tickers: the market tickers to subscribe to.
        signer: ``signer(method, path) -> headers`` for the signed WS handshake + REST.
        http: injectable async HTTP client for :func:`fetch_snapshot`.
        on_book: optional callback ``on_book(ticker, book)`` invoked after each applied delta
            (the downstream fill/snapshot sink); may be sync or async.
        ws_connect: injectable WS connector returning an async iterator of connections (the
            ``websockets.connect`` async-iterator idiom); a mock in tests. Defaults to
            ``websockets.connect``.
        demo: use the fixed demo hosts instead of prod (SSRF guard; both are constants).
        max_reconnects: optional bound on (re)connections — primarily for tests so a mock WS
            that always reconnects does not loop forever. ``None`` runs indefinitely.
    """
    ws_url, rest_host = _resolve_hosts(demo)
    connector = ws_connect or websockets.connect
    subscribe_cmd = _subscribe_command(tickers)
    # One book object per ticker; re-snapshot rebuilds it, never replaced wholesale, so the
    # downstream `on_book` reference stays stable across reconnects.
    books: dict[str, OrderBook] = {t: OrderBook(t) for t in tickers}

    connections = 0
    async for ws in connector(ws_url, additional_headers=dict(signer(_WS_METHOD, WS_PATH))):
        connections += 1
        try:
            # (Re)connect: re-subscribe AND force a REST re-snapshot BEFORE consuming deltas.
            await ws.send(subscribe_cmd)
            await _resnapshot_all(books, http, signer, tickers, rest_host=rest_host)

            async for raw in ws:
                msg = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
                ticker = _msg_ticker(msg, tickers)
                book = books[ticker]
                try:
                    apply(book, msg)
                except (SeqGap, CorrectnessError) as gap:
                    # The local book is unknown across the gap — discard + REST resync (D-02).
                    logger.warning(
                        "orderbook seq gap on %s (%s) — re-snapshotting via REST (D-02)",
                        ticker,
                        gap,
                    )
                    snapshot = await fetch_snapshot(http, signer, ticker, rest_host=rest_host)
                    apply(book, snapshot)
                    continue
                if on_book is not None:
                    result = on_book(ticker, book)
                    if isinstance(result, Awaitable):
                        await result
        except websockets.ConnectionClosed as exc:
            # EXPECTED transient: the iterator reconnects with backoff; structured fallback.
            logger.warning(
                "kalshi WS connection closed (%s) — reconnecting + re-snapshotting", exc
            )
        if max_reconnects is not None and connections >= max_reconnects:
            break


def _msg_ticker(msg: Mapping[str, Any], tickers: Sequence[str]) -> str:
    """Resolve the ticker a message belongs to (tolerant key; single-ticker shortcut)."""
    for key in ("market_ticker", "market_id", "ticker"):
        if key in msg:
            return msg[key]
    if len(tickers) == 1:
        return tickers[0]
    raise ValueError(
        f"orderbook message carries no ticker key and feed has multiple tickers: {msg!r}"
    )


__all__ = ["run_feed", "fetch_snapshot", "WS_URL_PROD", "WS_URL_DEMO", "REST_HOST_PROD", "REST_HOST_DEMO"]
