"""ingestion columns

Phase-2 (02-01) column-add migration. Extends the EXISTING ``forecasts`` and
``observations`` ledger tables with the ingestion payload columns (D-05/D-06/D-07) and
adds ``member`` to ``ix_forecasts_latest`` (between ``lead`` and ``available_at``, D-05).

This is **column-add only** — it never ``create_table``/``drop_table`` and never touches
the append-only trigger helpers. The Phase-1 ``BEFORE UPDATE/DELETE/TRUNCATE`` guards
installed by 0001 fire on the *unchanged* tables, so adding columns leaves them fully in
force (D-11, threat T-02-02). ``test_migration_0002`` asserts an UPDATE still raises.

Column types mirror ``weatherquant.db.models`` exactly so the migrated schema equals
``metadata.create_all``: DOUBLE PRECISION (``sa.Float``) for the temperature / coordinate
payloads, ``timestamptz`` for ``cycle``/``window_start``/``window_end``, ``smallint`` for
``member`` (NOT NULL, server default 0), and ``JSONB`` for the AFD/obs ``detail`` payload.

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-15

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '0002'
down_revision: Union[str, Sequence[str], None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add ingestion payload columns; extend ix_forecasts_latest with ``member``.

    Column-add only — no create_table/drop_table, no trigger DDL (the 0001 append-only
    guards persist on the unchanged tables, D-11).
    """
    # --- forecasts payload columns (D-05) -----------------------------------------
    op.add_column(
        'forecasts',
        sa.Column('member', sa.SmallInteger(), server_default='0', nullable=False),
    )
    op.add_column('forecasts', sa.Column('temp_kelvin', sa.Float(), nullable=True))
    op.add_column(
        'forecasts',
        sa.Column('cycle', sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column('forecasts', sa.Column('station_lat', sa.Float(), nullable=True))
    op.add_column('forecasts', sa.Column('station_lon', sa.Float(), nullable=True))
    op.add_column('forecasts', sa.Column('grid_distance_m', sa.Float(), nullable=True))

    # Extend the latest-row index to carry ``member`` (drop + recreate, D-05).
    op.drop_index('ix_forecasts_latest', table_name='forecasts')
    op.create_index(
        'ix_forecasts_latest',
        'forecasts',
        ['city', 'target_date', 'model', 'lead', 'member', 'available_at'],
        unique=False,
    )

    # --- observations payload columns (D-06/D-07) ---------------------------------
    op.add_column('observations', sa.Column('daily_high_f', sa.Float(), nullable=True))
    op.add_column(
        'observations',
        sa.Column('window_start', sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        'observations',
        sa.Column('window_end', sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column('observations', sa.Column('obs_count', sa.Integer(), nullable=True))
    op.add_column(
        'observations',
        sa.Column('detail', postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    """Reverse upgrade(): restore the 5-column index and drop the added columns."""
    # --- observations payload columns ---------------------------------------------
    op.drop_column('observations', 'detail')
    op.drop_column('observations', 'obs_count')
    op.drop_column('observations', 'window_end')
    op.drop_column('observations', 'window_start')
    op.drop_column('observations', 'daily_high_f')

    # --- restore the original 5-column ix_forecasts_latest (without ``member``) ----
    op.drop_index('ix_forecasts_latest', table_name='forecasts')
    op.create_index(
        'ix_forecasts_latest',
        'forecasts',
        ['city', 'target_date', 'model', 'lead', 'available_at'],
        unique=False,
    )

    # --- forecasts payload columns ------------------------------------------------
    op.drop_column('forecasts', 'grid_distance_m')
    op.drop_column('forecasts', 'station_lon')
    op.drop_column('forecasts', 'station_lat')
    op.drop_column('forecasts', 'cycle')
    op.drop_column('forecasts', 'temp_kelvin')
    op.drop_column('forecasts', 'member')
