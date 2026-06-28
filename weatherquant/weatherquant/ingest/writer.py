"""The SINGLE audited write path for the ledger payload tables (D-10 / D-11).

Every insert goes through one private helper that runs the skip-before-insert first (returning
0 on a match) and otherwise does a Core insert, raising unless ``rowcount == 1`` (an explicit
raise, not a stripped-under-``-O`` assert). No UPDATE/upsert path exists — the append-only
trigger would raise, so a correction is a later-``available_at`` INSERT. Requires a ``bind``
from :func:`get_engine` so ``preserve_rowcount`` holds (see docs/DECISIONS.md).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime

import sqlalchemy as sa
from sqlalchemy.engine import Connection

from weatherquant.db.models import fills, forecasts, market_snapshots, observations
from weatherquant.db.types import Bind, exec_bind
from weatherquant.ingest.errors import CorrectnessError
from weatherquant.ingest.idempotency import row_exists


class WriteIntegrityError(CorrectnessError, RuntimeError):
    """The single-row insert integrity contract was violated — ``rowcount != 1`` (D-11).

    A :class:`CorrectnessError` (and still a ``RuntimeError`` for back-compat) so the
    orchestrator re-raises this alarm loudly rather than swallowing a vanished row.
    """


def _insert_row(
    bind: Bind,
    table: sa.Table,
    natural_key: Mapping[str, object],
    content: Mapping[str, object],
    available_at: datetime,
) -> int:
    """Skip-before-insert one ledger row; return rowcount (1 inserted, 0 skipped) (D-10/D-11)."""

    def _do(conn: Connection) -> int:
        if row_exists(conn, table.name, natural_key, content):
            return 0  # identical row already in the ledger — skip (D-10), no UPDATE.
        values = {**natural_key, **content, "available_at": available_at}
        result = conn.execute(table.insert().values(**values))
        # Type the guard's input so a future ``rowcount -> int | None`` widening is caught by
        # mypy, not silently (TS-2, D-11).
        rowcount: int = result.rowcount
        # preserve_rowcount makes a single-row insert report 1 despite the implicit RETURNING id
        # (D-11). An explicit raise, NOT an assert: `python -O` strips asserts and would disable
        # the only check that the insert landed.
        if rowcount != 1:
            raise WriteIntegrityError(
                f"expected rowcount==1 inserting into {table.name}, got {rowcount}"
            )
        return rowcount

    with exec_bind(bind, write=True) as conn:
        return _do(conn)


def insert_forecast(
    bind: Bind,
    *,
    city: str,
    target_date: date,
    model: str,
    lead: int,
    member: int,
    temp_kelvin: float,
    cycle: datetime,
    station_lat: float,
    station_lon: float,
    grid_distance_m: float,
    available_at: datetime,
) -> int:
    """Insert one Kelvin forecast row through the single audited path (D-07/D-11).

    Natural key: ``(city, target_date, model, lead, member)`` — matches
    ``NATURAL_KEYS["forecasts"]``, the tuple ``latest()`` / DISTINCT-ON collapse on. ``cycle`` is
    an ADDITIONAL insert-time idempotency column (it joins the tuple passed to ``_insert_row``, so a
    new model ``cycle`` appends a distinct row rather than being skipped) — NOT part of the canonical
    natural key. Content: ``temp_kelvin`` + the station snap fields.

    Returns:
        ``1`` if a row was inserted, ``0`` if an identical row already existed (skip).
    """
    natural_key = {
        "city": city,
        "target_date": target_date,
        "model": model,
        "lead": lead,
        "member": member,
        "cycle": cycle,
    }
    content = {
        "temp_kelvin": temp_kelvin,
        "station_lat": station_lat,
        "station_lon": station_lon,
        "grid_distance_m": grid_distance_m,
    }
    return _insert_row(bind, forecasts, natural_key, content, available_at)


def insert_observation(
    bind: Bind,
    *,
    city: str,
    target_date: date,
    source: str,
    daily_high_f: float | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    obs_count: int | None = None,
    detail: Mapping[str, object] | None = None,
    available_at: datetime,
) -> int:
    """Insert one observation row through the SAME audited path as forecasts (D-10/D-11).

    Natural key: ``(city, target_date, source)`` — ``source='afd'`` slots in alongside the
    weather feeds. Content: the °F daily-high payload + the ``detail`` jsonb.

    Returns:
        ``1`` if a row was inserted, ``0`` if an identical row already existed (skip).
    """
    natural_key = {
        "city": city,
        "target_date": target_date,
        "source": source,
    }
    content = {
        "daily_high_f": daily_high_f,
        "window_start": window_start,
        "window_end": window_end,
        "obs_count": obs_count,
        "detail": detail,
    }
    return _insert_row(bind, observations, natural_key, content, available_at)


def insert_market_snapshot(
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
    """Insert one market-snapshot row through the SAME audited path (D-10/D-11/D-13).

    Natural key: ``(ticker, snapshot_for)``. Units are load-bearing: ``best_yes_bid``/
    ``best_no_bid`` in integer cents (the only side Kalshi quotes; ask is reflected elsewhere),
    ``mid`` in FLOAT-VALUED CENTS (CLV-consistent, no conversion), ``volume`` in WHOLE CONTRACTS
    (top-of-book size supporting the mid). Omitting ``volume`` persists NULL (back-compat).
    ``available_at`` is always a caller param — the real WS event time, never ``now()`` (D-08).

    Returns:
        ``1`` if a row was inserted, ``0`` if an identical row already existed (skip).
    """
    natural_key = {
        "ticker": ticker,
        "snapshot_for": snapshot_for,
    }
    content = {
        "best_yes_bid": best_yes_bid,
        "best_no_bid": best_no_bid,
        "mid": mid,
        "volume": volume,
        "seq": seq,
        "detail": detail,
    }
    return _insert_row(bind, market_snapshots, natural_key, content, available_at)


def insert_fill(
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
    """Insert one simulated-fill row through the SAME audited path (D-10/D-11/D-13).

    Natural key: ``(ticker, trade_id)``. Content: the execution payload (``price``/``fee`` in
    integer cents, ``event_time`` the real WS fill instant), the Phase-4 intent linkage
    (``bucket_prob``/``ev``/``kelly_stake``), and ``detail`` jsonb. ``available_at`` is always
    a caller param — the real WS event time, never ``now()`` (D-08).

    Returns:
        ``1`` if a row was inserted, ``0`` if an identical row already existed (skip).
    """
    # A maker rests at a real price; price in {None, 0, non-finite} is maker_queue_fill's
    # un-stamped placeholder (NaN) and would corrupt CLV as closing_mid - 0 (CORR-MED-4). Fail
    # loud with the money-path alarm rather than persist it. Taker fills are unaffected.
    if is_maker is True and (price is None or price == 0 or not math.isfinite(price)):
        raise WriteIntegrityError(
            f"refusing to persist a maker fill (ticker={ticker}, trade_id={trade_id}) with "
            f"price={price!r}: a maker rests at a real resting price; a None/0/non-finite price "
            "is the un-stamped maker_queue_fill placeholder and would corrupt CLV as "
            "closing_mid - 0 (CORR-MED-4)."
        )
    natural_key = {
        "ticker": ticker,
        "trade_id": trade_id,
    }
    content = {
        "side": side,
        "price": price,
        "count": count,
        "fee": fee,
        "is_maker": is_maker,
        "event_time": event_time,
        "bucket_prob": bucket_prob,
        "ev": ev,
        "kelly_stake": kelly_stake,
        "detail": detail,
    }
    return _insert_row(bind, fills, natural_key, content, available_at)


__all__ = [
    "WriteIntegrityError",
    "insert_fill",
    "insert_forecast",
    "insert_market_snapshot",
    "insert_observation",
]
