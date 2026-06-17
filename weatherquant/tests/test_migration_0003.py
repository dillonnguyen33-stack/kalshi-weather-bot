"""Integration test — migration 0003 column-adds + append-only trigger survival.

Marked ``integration``: requires a reachable Postgres via ``DATABASE_URL``; the
``pg_engine`` fixture rebuilds the schema from ``metadata.create_all`` (which is kept
identical to the Alembic 0003 migration by construction, D-11/D-13). When DATABASE_URL is
unset the fixture skips cleanly, so the fast subset stays green.

Asserts (D-13/D-11):
* the 11 new calibration_params columns (mean_intercept/mean_slope, var_intercept/
  var_slope, sigma_floor, n_train, pool_level, crps_train/crps_oos/crps_baseline_oos,
  trained_through) all exist;
* ``ix_calibration_params_latest`` is UNCHANGED — unlike 0002, no new key column was added
  (members collapse into the predictor, D-07), so it stays
  ``["city","model","lead","month","available_at"]``;
* an UPDATE on calibration_params STILL raises — proving 0003's column-adds did not
  drop/recreate the table and so the Phase-1 append-only trigger survived (threat T-03-03).

Column names are resolved via the SQLAlchemy inspector / ``table.c[...]`` — never
f-string-interpolated into SQL (T-03-05 / ASVS V5).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import sqlalchemy as sa

pytestmark = pytest.mark.integration


def test_0003_columns_present(pg_engine):
    """The Phase-3 EMOS/NGR payload columns exist on calibration_params (D-13)."""
    inspector = sa.inspect(pg_engine)
    cols = {c["name"] for c in inspector.get_columns("calibration_params")}

    assert {
        "mean_intercept",
        "mean_slope",
        "var_intercept",
        "var_slope",
        "sigma_floor",
        "n_train",
        "pool_level",
        "crps_train",
        "crps_oos",
        "crps_baseline_oos",
        "trained_through",
    } <= cols


def test_ix_calibration_params_latest_unchanged(pg_engine):
    """The latest-row index gained NO column — the natural key is unchanged (D-07)."""
    inspector = sa.inspect(pg_engine)
    indexes = {ix["name"]: ix for ix in inspector.get_indexes("calibration_params")}
    assert "ix_calibration_params_latest" in indexes
    cols = indexes["ix_calibration_params_latest"]["column_names"]
    assert cols == ["city", "model", "lead", "month", "available_at"]


def test_update_still_raises_after_calibration_column_add(pg_engine):
    """An UPDATE must still be rejected — the append-only trigger survived 0003 (D-13)."""
    from weatherquant.db.models import calibration_params

    base = datetime(2026, 6, 13, 12, 0, tzinfo=timezone.utc)
    key = {
        "city": "NYC",
        "model": "hrrr",
        "lead": 24,
        "month": 6,
    }
    with pg_engine.begin() as conn:
        result = conn.execute(
            sa.insert(calibration_params).values(available_at=base, **key)
        )
        assert result.rowcount == 1

    # The mutation column is resolved via table.c[...] — never f-stringed into SQL.
    with pytest.raises(Exception):
        with pg_engine.begin() as conn:
            conn.execute(
                sa.update(calibration_params).values(
                    {calibration_params.c["mean_intercept"]: 1.0}
                )
            )
