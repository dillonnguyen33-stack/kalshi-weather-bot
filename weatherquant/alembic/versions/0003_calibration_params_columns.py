"""calibration params columns

Phase-3 (03-02, D-13) column-add migration. Extends the EXISTING ``calibration_params``
ledger table with the EMOS/NGR payload columns — all in °F space (D-03) and all
``nullable=True`` so the column-add is non-breaking.

This is **column-add only** — it never creates or drops a table and never touches
the append-only trigger helpers. Unlike 0002 it also adds NO index column (the natural
key (city, model, lead, month) is unchanged — members collapse into the predictor, D-07),
so ``ix_calibration_params_latest`` is left fully intact. The Phase-1 ``BEFORE
UPDATE/DELETE/TRUNCATE`` guards installed by 0001 fire on the *unchanged* table, so adding
columns leaves them in force (D-11 / D-13, threat T-03-03). ``test_migration_0003`` asserts
an UPDATE still raises after this migration.

Column types mirror ``weatherquant.db.models`` exactly so the migrated schema equals
``metadata.create_all``: DOUBLE PRECISION (``sa.Float``) for the interpretable params
(``mean_intercept``/``mean_slope`` = a/b, ``var_intercept``/``var_slope`` = c/d,
``sigma_floor``) and the audit CRPS metrics (``crps_train``/``crps_oos``/
``crps_baseline_oos``); ``Integer`` for ``n_train``; ``Text`` for the ``pool_level``
provenance; and ``Date`` for the ``trained_through`` data cutoff.

Revision ID: 0003
Revises: 0002
Create Date: 2026-06-16

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '0003'
down_revision: Union[str, Sequence[str], None] = '0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add the calibration_params EMOS/NGR payload columns (D-13).

    Column-add only — no table create/drop, no trigger DDL, no index change (the
    0001 append-only guards persist on the unchanged table, D-11; the natural key gains no
    column so ix_calibration_params_latest is untouched).
    """
    # --- interpretable EMOS/NGR params, °F space (D-02/D-03) -----------------------
    op.add_column(
        'calibration_params',
        sa.Column('mean_intercept', sa.Float(), nullable=True),
    )
    op.add_column(
        'calibration_params',
        sa.Column('mean_slope', sa.Float(), nullable=True),
    )
    op.add_column(
        'calibration_params',
        sa.Column('var_intercept', sa.Float(), nullable=True),
    )
    op.add_column(
        'calibration_params',
        sa.Column('var_slope', sa.Float(), nullable=True),
    )
    op.add_column(
        'calibration_params',
        sa.Column('sigma_floor', sa.Float(), nullable=True),
    )
    op.add_column(
        'calibration_params',
        sa.Column('n_train', sa.Integer(), nullable=True),
    )
    op.add_column(
        'calibration_params',
        sa.Column('pool_level', sa.Text(), nullable=True),
    )
    # --- audit CRPS metrics (D-11) ------------------------------------------------
    op.add_column(
        'calibration_params',
        sa.Column('crps_train', sa.Float(), nullable=True),
    )
    op.add_column(
        'calibration_params',
        sa.Column('crps_oos', sa.Float(), nullable=True),
    )
    op.add_column(
        'calibration_params',
        sa.Column('crps_baseline_oos', sa.Float(), nullable=True),
    )
    # --- data cutoff so Phase 6 can re-derive a historical fit (D-13) --------------
    op.add_column(
        'calibration_params',
        sa.Column('trained_through', sa.Date(), nullable=True),
    )


def downgrade() -> None:
    """Reverse upgrade(): drop the 11 added columns in reverse order."""
    op.drop_column('calibration_params', 'trained_through')
    op.drop_column('calibration_params', 'crps_baseline_oos')
    op.drop_column('calibration_params', 'crps_oos')
    op.drop_column('calibration_params', 'crps_train')
    op.drop_column('calibration_params', 'pool_level')
    op.drop_column('calibration_params', 'n_train')
    op.drop_column('calibration_params', 'sigma_floor')
    op.drop_column('calibration_params', 'var_slope')
    op.drop_column('calibration_params', 'var_intercept')
    op.drop_column('calibration_params', 'mean_slope')
    op.drop_column('calibration_params', 'mean_intercept')
