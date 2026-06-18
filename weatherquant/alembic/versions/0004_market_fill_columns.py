"""market snapshot + fill columns

Phase-5 (05-00, D-11) column-add migration. Extends the EXISTING ``market_snapshots`` and
``fills`` ledger stubs with the Kalshi orderbook + simulated-fill payload columns — all
``nullable=True`` so the column-add is non-breaking.

This is **column-add only** — it never creates or drops a table and never touches the
append-only trigger helpers. Like 0003 it adds NO index column (the natural keys
(ticker, snapshot_for) / (ticker, trade_id) are unchanged), so ``ix_market_snapshots_latest``
and ``ix_fills_latest`` are left fully intact. The Phase-1 ``BEFORE UPDATE/DELETE/TRUNCATE``
guards installed by 0001 fire on the *unchanged* tables, so adding columns leaves them in
force (D-11, threat T-05-04). ``test_migration_0004`` asserts an UPDATE still raises on each
table after this migration.

Column types mirror ``weatherquant.db.models`` exactly so the migrated schema equals
``metadata.create_all``:

* ``market_snapshots``: ``best_yes_bid``/``best_no_bid`` (Integer cents), ``mid`` (Float),
  ``seq`` (BigInteger), ``detail`` (JSONB raw book payload).
* ``fills``: ``side`` (Text), ``price`` (Integer cents), ``count`` (Integer), ``fee``
  (Integer cents), ``is_maker`` (Boolean), ``event_time`` (timestamptz — the real WS fill
  time, D-08), the intent linkage ``bucket_prob``/``ev``/``kelly_stake`` (Float), and
  ``detail`` (JSONB raw trade payload).

Revision ID: 0004
Revises: 0003
Create Date: 2026-06-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '0004'
down_revision: Union[str, Sequence[str], None] = '0003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the market_snapshots + fills Phase-5 payload columns (D-11).

    Column-add only — no table create/drop, no trigger DDL, no index change (the
    0001 append-only guards persist on the unchanged tables, D-11; the natural keys gain
    no column so ix_market_snapshots_latest / ix_fills_latest are untouched).
    """
    # --- market_snapshots: Kalshi orderbook payload (cents + derived mid + WS seq) ---
    op.add_column(
        'market_snapshots',
        sa.Column('best_yes_bid', sa.Integer(), nullable=True),
    )
    op.add_column(
        'market_snapshots',
        sa.Column('best_no_bid', sa.Integer(), nullable=True),
    )
    op.add_column(
        'market_snapshots',
        sa.Column('mid', sa.Float(), nullable=True),
    )
    op.add_column(
        'market_snapshots',
        sa.Column('seq', sa.BigInteger(), nullable=True),
    )
    op.add_column(
        'market_snapshots',
        sa.Column('detail', postgresql.JSONB(), nullable=True),
    )
    # --- fills: simulated-execution payload + intent linkage (D-08) ------------------
    op.add_column(
        'fills',
        sa.Column('side', sa.Text(), nullable=True),
    )
    op.add_column(
        'fills',
        sa.Column('price', sa.Integer(), nullable=True),
    )
    op.add_column(
        'fills',
        sa.Column('count', sa.Integer(), nullable=True),
    )
    op.add_column(
        'fills',
        sa.Column('fee', sa.Integer(), nullable=True),
    )
    op.add_column(
        'fills',
        sa.Column('is_maker', sa.Boolean(), nullable=True),
    )
    op.add_column(
        'fills',
        sa.Column('event_time', sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        'fills',
        sa.Column('bucket_prob', sa.Float(), nullable=True),
    )
    op.add_column(
        'fills',
        sa.Column('ev', sa.Float(), nullable=True),
    )
    op.add_column(
        'fills',
        sa.Column('kelly_stake', sa.Float(), nullable=True),
    )
    op.add_column(
        'fills',
        sa.Column('detail', postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    """Reverse upgrade(): drop the added columns in reverse order."""
    op.drop_column('fills', 'detail')
    op.drop_column('fills', 'kelly_stake')
    op.drop_column('fills', 'ev')
    op.drop_column('fills', 'bucket_prob')
    op.drop_column('fills', 'event_time')
    op.drop_column('fills', 'is_maker')
    op.drop_column('fills', 'fee')
    op.drop_column('fills', 'count')
    op.drop_column('fills', 'price')
    op.drop_column('fills', 'side')
    op.drop_column('market_snapshots', 'detail')
    op.drop_column('market_snapshots', 'seq')
    op.drop_column('market_snapshots', 'mid')
    op.drop_column('market_snapshots', 'best_no_bid')
    op.drop_column('market_snapshots', 'best_yes_bid')
