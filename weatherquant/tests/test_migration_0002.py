"""Integration test — migration 0002 column-adds + append-only trigger survival.

Marked ``integration``: requires a reachable Postgres via ``DATABASE_URL``; the
``pg_engine`` fixture rebuilds the schema from ``metadata.create_all`` (which is kept
identical to the Alembic 0002 migration by construction, D-11). When DATABASE_URL is unset
the fixture skips cleanly, so the fast subset stays green.

Asserts (D-05/D-06/D-07/D-11):
* the new forecasts columns (member, temp_kelvin, cycle, station_lat/lon, grid_distance_m)
  and observations columns (daily_high_f, window_start/end, obs_count, detail) all exist;
* ``ix_forecasts_latest`` includes ``member``;
* an UPDATE on forecasts STILL raises — proving 0002's column-adds did not drop/recreate
  the table and so the Phase-1 append-only trigger survived (threat T-02-02).

Column names are resolved via the SQLAlchemy inspector / ``table.c[...]`` — never
f-string-interpolated into SQL.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
import sqlalchemy as sa

pytestmark = pytest.mark.integration


def test_0002_columns_present(pg_engine):
    """The Phase-2 ingestion columns exist on forecasts + observations."""
    inspector = sa.inspect(pg_engine)
    forecast_cols = {c["name"] for c in inspector.get_columns("forecasts")}
    obs_cols = {c["name"] for c in inspector.get_columns("observations")}

    assert {
        "member",
        "temp_kelvin",
        "cycle",
        "station_lat",
        "station_lon",
        "grid_distance_m",
    } <= forecast_cols
    assert {
        "daily_high_f",
        "window_start",
        "window_end",
        "obs_count",
        "detail",
    } <= obs_cols


def test_ix_forecasts_latest_includes_member(pg_engine):
    """ix_forecasts_latest carries ``member`` between ``lead`` and ``available_at``."""
    inspector = sa.inspect(pg_engine)
    indexes = {ix["name"]: ix for ix in inspector.get_indexes("forecasts")}
    assert "ix_forecasts_latest" in indexes
    cols = indexes["ix_forecasts_latest"]["column_names"]
    assert "member" in cols
    assert cols == ["city", "target_date", "model", "lead", "member", "available_at"]


def test_update_still_raises_after_column_add(pg_engine):
    """An UPDATE must still be rejected — the append-only trigger survived 0002 (D-11)."""
    from weatherquant.db.models import forecasts

    base = datetime(2026, 6, 13, 12, 0, tzinfo=timezone.utc)
    key = {
        "city": "NYC",
        "target_date": date(2026, 6, 14),
        "model": "hrrr",
        "lead": 24,
    }
    with pg_engine.begin() as conn:
        result = conn.execute(
            sa.insert(forecasts).values(available_at=base, **key)
        )
        assert result.rowcount == 1

    # The mutation column is resolved via table.c[...] — never f-stringed into SQL.
    with pytest.raises(Exception):
        with pg_engine.begin() as conn:
            conn.execute(
                sa.update(forecasts).values({forecasts.c["temp_kelvin"]: 300.0})
            )
