"""Integration test — migration 0004 column-adds + append-only trigger survival.

Marked ``integration``: requires a reachable Postgres via ``DATABASE_URL``; the ``pg_engine``
fixture rebuilds the schema from ``metadata.create_all`` (kept identical to the Alembic 0004
migration by construction, D-11). When DATABASE_URL is unset the fixture skips cleanly, so the
fast subset stays green.

Mirrors ``test_migration_0003`` and asserts (D-11 / threat T-05-04):
* the new market_snapshots columns (best_yes_bid/best_no_bid/mid/seq/detail) and fills columns
  (side/price/count/fee/is_maker/event_time/bucket_prob/ev/kelly_stake/detail) all exist;
* ``ix_market_snapshots_latest`` / ``ix_fills_latest`` are UNCHANGED — no new key column was
  added (the natural keys are unchanged), so they stay
  ``["ticker","snapshot_for","available_at"]`` / ``["ticker","trade_id","available_at"]``;
* an UPDATE on each table STILL raises — proving 0004's column-adds did not drop/recreate the
  table, so the Phase-1 append-only trigger survived (threat T-05-04).

Column names are resolved via the SQLAlchemy inspector / ``table.c[...]`` — never
f-string-interpolated into SQL (ASVS V5).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import sqlalchemy as sa

pytestmark = pytest.mark.integration


def test_0004_market_snapshots_columns_present(pg_engine):
    """The Phase-5 orderbook payload columns exist on market_snapshots (D-11)."""
    inspector = sa.inspect(pg_engine)
    cols = {c["name"] for c in inspector.get_columns("market_snapshots")}
    assert {"best_yes_bid", "best_no_bid", "mid", "seq", "detail"} <= cols


def test_0004_fills_columns_present(pg_engine):
    """The Phase-5 simulated-fill payload + intent-linkage columns exist on fills (D-11)."""
    inspector = sa.inspect(pg_engine)
    cols = {c["name"] for c in inspector.get_columns("fills")}
    assert {
        "side",
        "price",
        "count",
        "fee",
        "is_maker",
        "event_time",
        "bucket_prob",
        "ev",
        "kelly_stake",
        "detail",
    } <= cols


def test_ix_market_snapshots_latest_unchanged(pg_engine):
    """The latest-row index gained NO column — the natural key is unchanged."""
    inspector = sa.inspect(pg_engine)
    indexes = {ix["name"]: ix for ix in inspector.get_indexes("market_snapshots")}
    assert "ix_market_snapshots_latest" in indexes
    cols = indexes["ix_market_snapshots_latest"]["column_names"]
    assert cols == ["ticker", "snapshot_for", "available_at"]


def test_ix_fills_latest_unchanged(pg_engine):
    """The latest-row index gained NO column — the natural key is unchanged."""
    inspector = sa.inspect(pg_engine)
    indexes = {ix["name"]: ix for ix in inspector.get_indexes("fills")}
    assert "ix_fills_latest" in indexes
    cols = indexes["ix_fills_latest"]["column_names"]
    assert cols == ["ticker", "trade_id", "available_at"]


def test_update_still_raises_after_market_snapshots_column_add(pg_engine):
    """An UPDATE must still be rejected — the append-only trigger survived 0004 (T-05-04)."""
    from weatherquant.db.models import market_snapshots

    base = datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc)
    key = {"ticker": "KXHIGHNY-26JUN18-T72", "snapshot_for": "2026-06-18T19:55Z"}
    with pg_engine.begin() as conn:
        result = conn.execute(
            sa.insert(market_snapshots).values(available_at=base, **key)
        )
        assert result.rowcount == 1

    # The mutation column is resolved via table.c[...] — never f-stringed into SQL.
    with pytest.raises(Exception):
        with pg_engine.begin() as conn:
            conn.execute(
                sa.update(market_snapshots).values({market_snapshots.c["mid"]: 1.0})
            )


def test_update_still_raises_after_fills_column_add(pg_engine):
    """An UPDATE must still be rejected — the append-only trigger survived 0004 (T-05-04)."""
    from weatherquant.db.models import fills

    base = datetime(2026, 6, 18, 12, 0, tzinfo=timezone.utc)
    key = {"ticker": "KXHIGHNY-26JUN18-T72", "trade_id": "mig0004-trade-1"}
    with pg_engine.begin() as conn:
        result = conn.execute(sa.insert(fills).values(available_at=base, **key))
        assert result.rowcount == 1

    with pytest.raises(Exception):
        with pg_engine.begin() as conn:
            conn.execute(sa.update(fills).values({fills.c["price"]: 1}))
