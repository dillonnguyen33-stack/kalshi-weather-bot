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

import email.utils
import json
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import httpx
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
        # Validate the level is a 2-element [price, count] sequence BEFORE unpacking (mirror
        # book._coerce_levels' len-2 guard) — a malformed level raises a descriptive ValueError,
        # never a coerced/fabricated level or an opaque "not enough values to unpack" (HIGH-2).
        if isinstance(level, (str, bytes, Mapping)):
            raise ValueError(f"orderbook_fp level must be [price, count], got {level!r}")
        items = list(level)
        if len(items) != 2:
            raise ValueError(
                f"orderbook_fp level must be exactly [price, count], got {items!r}"
            )
        price_d, count = items
        levels.append([_cents(price_d), round(float(count))])
    return levels


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
            try:
                await _resnapshot_all(books, http, signer, tickers, rest_host=rest_host)
            except httpx.HTTPError as exc:
                # A transient REST blip on (re)connection must NOT kill the feed: degrade to a
                # reconnect so the `async for ws` iterator re-handshakes + retries the resync
                # (WR-01). fetch_snapshot already logged + re-raised FAIL-LOUD at its boundary.
                logger.warning(
                    "[market error] on-reconnect re-snapshot failed (%s) — forcing WS reconnect",
                    exc,
                )
                continue

            async for raw in ws:
                msg = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else raw
                ticker = _msg_ticker(msg, tickers)
                book = books[ticker]
                try:
                    apply(book, msg)
                except (SeqGap, CorrectnessError) as gap:
                    # The local book is unknown across the gap — discard + REST resync (D-02).
                    logger.warning(
                        "[orderbook error] seq gap on %s (%s) — re-snapshotting via REST (D-02)",
                        ticker,
                        gap,
                    )
                    try:
                        snapshot = await fetch_snapshot(http, signer, ticker, rest_host=rest_host)
                    except httpx.HTTPError as exc:
                        # A transient REST blip during the exact moment the book must re-anchor
                        # must NOT crash the feed: break the delta loop so the `async for ws`
                        # iterator reconnects + re-snapshots (WR-01).
                        logger.warning(
                            "[market error] re-snapshot of %s failed (%s) — forcing WS reconnect",
                            ticker,
                            exc,
                        )
                        break
                    apply(book, snapshot)
                    # Surface the freshly re-anchored book to the sink BEFORE continuing —
                    # otherwise every REST-resync book state (precisely the states most likely
                    # to fall inside a CLV closing window after a disruption) is silently
                    # dropped from persistence (WR-02 / PAP-04 cadence sufficiency).
                    if on_book is not None:
                        result = on_book(ticker, book)
                        if isinstance(result, Awaitable):
                            await result
                    continue
                if on_book is not None:
                    result = on_book(ticker, book)
                    if isinstance(result, Awaitable):
                        await result
        except websockets.ConnectionClosed as exc:
            # EXPECTED transient: the iterator reconnects with backoff; structured fallback.
            logger.warning(
                "[ws error] kalshi WS connection closed (%s) — reconnecting + re-snapshotting",
                exc,
            )
        if max_reconnects is not None and connections >= max_reconnects:
            break


def _msg_ticker(msg: Mapping[str, Any], tickers: Sequence[str]) -> str:
    """Resolve the ticker a message belongs to (tolerant key; single-ticker shortcut).

    The ticker value is UNTRUSTED WS/REST JSON that immediately keys ``books[ticker]`` in
    ``run_feed`` — the trust boundary (TS-1). A present ticker key whose value is not a str
    (e.g. an int ``market_id``) FAILS LOUD here rather than silently mis-keying or
    KeyErroring another ticker's book; the ``-> str`` annotation is now actually enforced.
    """
    for key in ("market_ticker", "market_id", "ticker"):
        if key in msg:
            value = msg[key]
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
