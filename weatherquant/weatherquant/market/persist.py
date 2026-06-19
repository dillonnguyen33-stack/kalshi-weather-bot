"""Thin snapshot/fill write+read adapter over the one audited ledger path (PAP-03/PAP-04).

This module is deliberately THIN: it contains NO Core insert of its own. Every write
delegates to :func:`weatherquant.ingest.writer.insert_market_snapshot` /
:func:`~weatherquant.ingest.writer.insert_fill` — the single audited append-only path
(skip-before-insert idempotency, the ``rowcount == 1`` integrity raise, ``available_at`` a
caller param, NO UPDATE/upsert; D-10/D-11/D-13). Every read delegates to
:func:`weatherquant.db.queries.latest` with the FULL natural key (an under-specified key is
rejected there, the WR-02 trap).

The point of the adapter is a small, market-flavoured surface (``persist_snapshot`` /
``persist_fill`` / ``latest_snapshots``) the ``run_paper`` CLI orchestrates, so the I/O edge
never re-derives the ledger contract or hand-rolls a Core insert (threat T-05-15).

It is NOT pure — it touches the DB bind — but it does so only through the audited writer +
``queries.latest`` and imports no ``websockets``/``cryptography``/SDK (it is the persistence
seam, not the live feed).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime

from sqlalchemy.engine import Connection, Engine, RowMapping

from weatherquant.db import queries
from weatherquant.ingest.writer import insert_fill, insert_market_snapshot

Bind = Engine | Connection


def persist_snapshot(
    bind: Bind,
    *,
    ticker: str,
    snapshot_for: str,
    best_yes_bid: int | None = None,
    best_no_bid: int | None = None,
    mid: float | None = None,
    volume: int | None = None,
    seq: int | None = None,
    detail: Mapping[str, object] | None = None,
    available_at: datetime,
) -> int:
    """Persist one market snapshot via the audited writer (no Core insert here, D-13).

    Delegates verbatim to :func:`weatherquant.ingest.writer.insert_market_snapshot`.
    ``mid`` is FLOAT-VALUED CENTS (CR-01) and ``volume`` is the per-snapshot book-liquidity
    signal in whole contracts (WR-01) — both threaded straight through. ``available_at`` is
    the REAL WS event time (D-08) — passed straight through, never ``now()``.

    Returns:
        ``1`` if a row was inserted, ``0`` if an identical row already existed (skip).
    """
    return insert_market_snapshot(
        bind,
        ticker=ticker,
        snapshot_for=snapshot_for,
        best_yes_bid=best_yes_bid,
        best_no_bid=best_no_bid,
        mid=mid,
        volume=volume,
        seq=seq,
        detail=detail,
        available_at=available_at,
    )


def persist_fill(
    bind: Bind,
    *,
    ticker: str,
    trade_id: str,
    side: str | None = None,
    price: int | None = None,
    count: int | None = None,
    fee: int | None = None,
    is_maker: bool | None = None,
    event_time: datetime | None = None,
    bucket_prob: float | None = None,
    ev: float | None = None,
    kelly_stake: float | None = None,
    detail: Mapping[str, object] | None = None,
    available_at: datetime,
) -> int:
    """Persist one simulated fill via the audited writer (no Core insert here, D-13).

    Delegates verbatim to :func:`weatherquant.ingest.writer.insert_fill`. ``available_at``
    and ``event_time`` are caller params (the real WS event time, D-08), never ``now()``.

    Returns:
        ``1`` if a row was inserted, ``0`` if an identical row already existed (skip).
    """
    return insert_fill(
        bind,
        ticker=ticker,
        trade_id=trade_id,
        side=side,
        price=price,
        count=count,
        fee=fee,
        is_maker=is_maker,
        event_time=event_time,
        bucket_prob=bucket_prob,
        ev=ev,
        kelly_stake=kelly_stake,
        detail=detail,
        available_at=available_at,
    )


def latest_snapshots(bind: Bind, ticker: str) -> list[RowMapping]:
    """Return the latest ``market_snapshots`` row per ``snapshot_for`` for ``ticker``.

    Reads through :func:`weatherquant.db.queries.latest` with the table's FULL canonical
    natural key ``(ticker, snapshot_for)`` (resolved from ``NATURAL_KEYS``; an
    under-specified key would be rejected — WR-02). The ``where={"ticker": ticker}`` scope
    filter is applied as a bound parameter before the ``DISTINCT ON``, so this returns one
    row per distinct ``snapshot_for`` for the ticker, each the newest by ``available_at``.

    These are the rows the CLV closing-window selection (``market.clv``) consumes after
    settlement.
    """
    return queries.latest(bind, "market_snapshots", where={"ticker": ticker})


__all__ = ["persist_snapshot", "persist_fill", "latest_snapshots"]
