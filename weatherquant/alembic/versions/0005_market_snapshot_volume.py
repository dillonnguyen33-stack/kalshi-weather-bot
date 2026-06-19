"""market snapshot volume column

Phase-5 gap-closure (05-06, WR-01) additive column-add migration. Adds a single
``volume`` column to the EXISTING ``market_snapshots`` ledger table so the derived CLV
(``weatherquant.market.clv.vol_weighted_mid``) can weight the closing mid by a REAL
persisted per-snapshot liquidity signal instead of a hand-wired fixture key that no
production row carried (the WR-01 KeyError-on-every-production-row defect).

This is **column-add only** — mirroring 0004 EXACTLY — so it never creates or drops a
table and never touches the append-only trigger helpers. It adds NO index column (the
natural key (ticker, snapshot_for) is unchanged), so ``ix_market_snapshots_latest`` is
left fully intact. The Phase-1 ``BEFORE UPDATE/DELETE/TRUNCATE`` guards installed by 0001
fire on the *unchanged* table, so adding a column leaves them in force (D-11, threat
T-05-04, T-05-23). Existing rows get a NULL ``volume`` (non-breaking; ``nullable=True``).

``volume`` is ``Integer`` (not ``Float``): traded/resting contract volume is a
whole-contract count. The half-cent precision concern lives on ``mid`` (which stays
``Float`` cents), not on ``volume``.

The column type mirrors ``weatherquant.db.models.market_snapshots.volume`` exactly so the
migrated schema equals ``metadata.create_all``.

Revision ID: 0005
Revises: 0004
Create Date: 2026-06-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0005'
down_revision: Union[str, Sequence[str], None] = '0004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the market_snapshots.volume column (WR-01).

    Column-add only — no table create/drop, no trigger DDL, no index change (the
    0001 append-only guards persist on the unchanged table, T-05-23; the natural key gains
    no column so ix_market_snapshots_latest is untouched). Integer (whole-contract count);
    nullable so existing rows are non-breaking.
    """
    op.add_column(
        'market_snapshots',
        sa.Column('volume', sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    """Reverse upgrade(): drop the added column."""
    op.drop_column('market_snapshots', 'volume')
