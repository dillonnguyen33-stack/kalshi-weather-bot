"""insert_market_snapshot / insert_fill audited write path (D-13, 05-04 GREEN).

The two new audited writers extend ``weatherquant.ingest.writer`` and reuse its ``_insert_row``
helper verbatim: skip-before-insert (``row_exists``) returns a no-op on a duplicate natural-key
+ content row (idempotency, D-10), and an explicit ``rowcount != 1`` raise
(``WriteIntegrityError``, survives ``python -O``, D-11). ``available_at`` is ALWAYS a caller
param (= the real WS event time, never ``now()``). There is NO UPDATE/upsert path — a
correction is a later-``available_at`` INSERT, and an attempted UPDATE is raised loudly by the
append-only trigger.

Marked ``integration`` (the assertions exercise the DB via ``pg_conn``). 05-04 implements
``insert_market_snapshot`` / ``insert_fill`` and flips these GREEN.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import sqlalchemy as sa

writer = pytest.importorskip("weatherquant.ingest.writer")

from weatherquant.db.models import fills, market_snapshots  # noqa: E402

pytestmark = pytest.mark.integration

_HAS_MARKET_WRITERS = hasattr(writer, "insert_market_snapshot") and hasattr(
    writer, "insert_fill"
)
_skip_until_0504 = pytest.mark.skipif(
    not _HAS_MARKET_WRITERS,
    reason="RED — 05-04 adds insert_market_snapshot / insert_fill to the writer",
)

_T0 = datetime(2026, 6, 18, 19, 55, tzinfo=timezone.utc)
_T1 = datetime(2026, 6, 18, 19, 56, tzinfo=timezone.utc)


@_skip_until_0504
def test_insert_market_snapshot_rowcount_one(pg_conn):
    """A single snapshot insert lands exactly one row (rowcount==1 contract, D-11)."""
    rc = writer.insert_market_snapshot(
        pg_conn,
        ticker="KXHIGHNY-26JUN18-T72",
        snapshot_for="2026-06-18T19:55Z",
        best_yes_bid=50,
        best_no_bid=48,
        mid=0.51,
        seq=101,
        detail={"raw": "book"},
        available_at=_T0,
    )
    assert rc == 1


@_skip_until_0504
def test_insert_market_snapshot_idempotent(pg_conn):
    """Re-inserting an identical snapshot is a no-op skip; a changed payload appends (D-10)."""
    kw = dict(
        ticker="KXHIGHNY-26JUN18-T72",
        snapshot_for="2026-06-18T19:55Z",
        best_yes_bid=50,
        best_no_bid=48,
        mid=0.51,
        seq=101,
        detail={"raw": "book"},
        available_at=_T0,
    )
    assert writer.insert_market_snapshot(pg_conn, **kw) == 1
    # Identical natural key + content + available_at → skip-before-insert returns 0.
    assert writer.insert_market_snapshot(pg_conn, **kw) == 0
    # A changed payload (new mid) is a fresh append → 1.
    assert writer.insert_market_snapshot(pg_conn, **{**kw, "mid": 0.52}) == 1


@_skip_until_0504
def test_insert_market_snapshot_update_raises(pg_conn):
    """The append-only trigger raises on any UPDATE — corrections are later-available_at INSERTs."""
    writer.insert_market_snapshot(
        pg_conn,
        ticker="KXHIGHNY-26JUN18-T72",
        snapshot_for="2026-06-18T19:55Z",
        mid=0.51,
        available_at=_T0,
    )
    with pytest.raises(Exception):  # noqa: B017 — the per-table append-only trigger RAISEs
        pg_conn.execute(
            sa.update(market_snapshots)
            .where(market_snapshots.c.ticker == "KXHIGHNY-26JUN18-T72")
            .values(mid=0.99)
        )


@_skip_until_0504
def test_insert_fill_rowcount_one(pg_conn):
    """A single fill insert lands exactly one row (rowcount==1 contract, D-11)."""
    rc = writer.insert_fill(
        pg_conn,
        ticker="KXHIGHNY-26JUN18-T72",
        trade_id="t-0001",
        side="yes",
        price=50,
        count=100,
        fee=2,
        is_maker=False,
        event_time=_T0,
        bucket_prob=0.62,
        ev=0.05,
        kelly_stake=0.02,
        detail={"raw": "trade"},
        available_at=_T0,
    )
    assert rc == 1


@_skip_until_0504
def test_insert_fill_idempotent(pg_conn):
    """Re-inserting an identical fill is a no-op skip; a changed payload appends (D-10)."""
    kw = dict(
        ticker="KXHIGHNY-26JUN18-T72",
        trade_id="t-0001",
        side="yes",
        price=50,
        count=100,
        fee=2,
        is_maker=False,
        event_time=_T0,
        bucket_prob=0.62,
        ev=0.05,
        kelly_stake=0.02,
        detail={"raw": "trade"},
        available_at=_T0,
    )
    assert writer.insert_fill(pg_conn, **kw) == 1
    assert writer.insert_fill(pg_conn, **kw) == 0
    # A changed payload (different count) is a fresh append → 1.
    assert writer.insert_fill(pg_conn, **{**kw, "count": 130}) == 1


@_skip_until_0504
def test_insert_fill_update_raises(pg_conn):
    """The append-only trigger raises on any UPDATE to a fill row (D-10)."""
    writer.insert_fill(
        pg_conn,
        ticker="KXHIGHNY-26JUN18-T72",
        trade_id="t-0001",
        price=50,
        available_at=_T0,
    )
    with pytest.raises(Exception):  # noqa: B017 — append-only trigger RAISEs on UPDATE
        pg_conn.execute(
            sa.update(fills)
            .where(fills.c.trade_id == "t-0001")
            .values(price=99)
        )
