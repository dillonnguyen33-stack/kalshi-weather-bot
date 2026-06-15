"""RED integration tests for the append-only Postgres ledger (SYS-01 / D-10, D-11, D-12).

Marked ``integration`` — they require a reachable Postgres via ``DATABASE_URL``
(``postgresql+psycopg://``). When DATABASE_URL is unset the ``pg_engine`` fixture skips
cleanly, so the fast subset stays green. When set, these run against the schema built by
plan 01-03's Alembic migration / ``metadata.create_all``.

Imports ``weatherquant.db.models`` and ``weatherquant.db.queries`` (plan 01-03) — RED
until then (ImportError). Behavior under test:
* each of the 5 tables (forecasts, observations, calibration_params, market_snapshots,
  fills) accepts an INSERT;
* ``latest(...)`` returns the newest row per natural key by ``available_at DESC, id DESC``
  (Pitfall 5 deterministic tiebreaker);
* the append-only contract: corrections are new inserts, not UPDATE/DELETE (D-10).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

pytestmark = pytest.mark.integration

LEDGER_TABLES = [
    "forecasts",
    "observations",
    "calibration_params",
    "market_snapshots",
    "fills",
]


def _import_db():
    """Deferred import — weatherquant.db lands in plan 01-03 (RED: ImportError)."""
    from weatherquant.db import models, queries

    return models, queries


def test_metadata_defines_five_tables():
    models, _queries = _import_db()
    names = set(models.metadata.tables.keys())
    for t in LEDGER_TABLES:
        assert t in names, f"missing ledger table: {t}"


@pytest.mark.parametrize("table_name", LEDGER_TABLES)
def test_each_table_accepts_insert(pg_engine, table_name):
    import sqlalchemy as sa

    models, _queries = _import_db()
    table = models.metadata.tables[table_name]
    now = datetime.now(timezone.utc)
    # Insert with the columns we know are contractually present; defaults cover the rest.
    values = {"available_at": now}
    for col in table.columns:
        if col.name in values or col.primary_key:
            continue
        if not col.nullable and col.default is None and col.server_default is None:
            values[col.name] = _sample_value(col)
    with pg_engine.begin() as conn:
        result = conn.execute(sa.insert(table).values(**values))
        assert result.rowcount == 1


def test_latest_returns_newest_by_available_at_then_id(pg_engine):
    import sqlalchemy as sa

    models, queries = _import_db()
    forecasts = models.metadata.tables["forecasts"]
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    key = {"city": "NYC", "target_date": "2025-01-15", "model": "gfs", "lead": 24}
    with pg_engine.begin() as conn:
        # two rows, same natural key, increasing available_at
        conn.execute(sa.insert(forecasts).values(available_at=base, **key))
        conn.execute(
            sa.insert(forecasts).values(available_at=base + timedelta(hours=1), **key)
        )
    rows = queries.latest(pg_engine, "forecasts", ["city", "target_date", "model", "lead"])
    matching = [r for r in rows if r["city"] == "NYC" and r["lead"] == 24]
    assert len(matching) == 1, "latest must collapse to one row per natural key"
    assert matching[0]["available_at"] == base + timedelta(hours=1)


def test_append_only_no_update(pg_engine):
    """D-10: the ledger is insert-only. An attempted UPDATE must be rejected (by DB role
    or convention guard). Corrections are new inserts with a later available_at."""
    import sqlalchemy as sa

    models, _queries = _import_db()
    forecasts = models.metadata.tables["forecasts"]
    with pytest.raises(Exception):
        with pg_engine.begin() as conn:
            conn.execute(sa.update(forecasts).values(city="MUTATED"))


@pytest.mark.parametrize("table_name", LEDGER_TABLES)
def test_append_only_no_truncate(pg_engine, table_name):
    """D-10: TRUNCATE must be rejected on every ledger table. The per-row UPDATE/DELETE
    trigger does NOT fire on TRUNCATE, so a dedicated BEFORE TRUNCATE statement-level
    trigger guards against a TRUNCATE silently wiping the append-only ledger."""
    import sqlalchemy as sa

    _import_db()
    with pytest.raises(Exception):
        with pg_engine.begin() as conn:
            conn.execute(sa.text(f'TRUNCATE "{table_name}"'))


def _sample_value(col):
    import sqlalchemy as sa

    t = col.type
    if isinstance(t, sa.Integer):
        return 1
    if isinstance(t, (sa.Text, sa.String)):
        return "x"
    if isinstance(t, sa.Date):
        return "2025-01-15"
    return "x"
