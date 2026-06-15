"""SQLAlchemy Core schema for the append-only weatherquant ledger (D-10, D-11, D-12).

This module defines a single :data:`metadata` object and the FIVE skeletal ledger
tables every later phase writes to:

    forecasts · observations · calibration_params · market_snapshots · fills

The shared contract (baked in NOW so phases 2–6 extend columns via Alembic rather than
redefining tables — D-11):

* **id** — surrogate serial primary key (``BigInteger, Identity()``). The PK is the
  surrogate, NOT the natural key.
* **natural key** — the business identity columns (e.g. city/target_date/model/lead for
  forecasts). They are ``nullable=False`` and covered by an ``ix_<table>_latest`` index,
  but are **NEVER** declared UNIQUE: the ledger is append-only, so many rows may share a
  natural key (each a successive point-in-time observation of the same fact).
* **available_at** — ``TIMESTAMP(timezone=True), nullable=False``. Point-in-time-of-
  knowledge (D-12): the moment the datum became available to the system, set by the
  writer. Never back-dated to the datum's nominal timestamp — that would destroy the
  no-look-ahead integrity Phase 6's walk-forward depends on.

**Insert-only — no UPDATE, no DELETE, ever (D-10).** Corrections are new inserts with a
later ``available_at``. This is enforced structurally: a per-table PostgreSQL rule
(attached to :data:`metadata` via DDL events, so it fires under both ``create_all`` and
the Alembic migration) raises on any UPDATE or DELETE. The "latest" read idiom lives in
:mod:`weatherquant.db.queries` (``DISTINCT ON (natural_key) ORDER BY ..., available_at
DESC, id DESC``).

Payload columns are deliberately omitted here — they are added by phases 2–6 via Alembic
migrations. SQLAlchemy Core only (no ORM declarative base). The legacy psycopg-2 driver
is never referenced; the engine uses the ``postgresql+psycopg://`` (v3) dialect.
"""

from __future__ import annotations

import sqlalchemy as sa

metadata = sa.MetaData()


def _id_column() -> sa.Column:
    """Surrogate serial primary key shared by every ledger table."""
    return sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True)


def _available_at_column() -> sa.Column:
    """Point-in-time-of-knowledge timestamptz shared by every ledger table (D-12)."""
    return sa.Column("available_at", sa.TIMESTAMP(timezone=True), nullable=False)


# --- forecasts: natural key = city, target_date, model, lead -----------------------
forecasts = sa.Table(
    "forecasts",
    metadata,
    _id_column(),
    sa.Column("city", sa.Text, nullable=False),
    sa.Column("target_date", sa.Date, nullable=False),
    sa.Column("model", sa.Text, nullable=False),
    sa.Column("lead", sa.Integer, nullable=False),
    _available_at_column(),
    sa.Index(
        "ix_forecasts_latest",
        "city",
        "target_date",
        "model",
        "lead",
        "available_at",
    ),
)

# --- observations: natural key = city, target_date (LST settlement day), source ----
observations = sa.Table(
    "observations",
    metadata,
    _id_column(),
    sa.Column("city", sa.Text, nullable=False),
    sa.Column("target_date", sa.Date, nullable=False),
    sa.Column("source", sa.Text, nullable=False),
    _available_at_column(),
    sa.Index(
        "ix_observations_latest",
        "city",
        "target_date",
        "source",
        "available_at",
    ),
)

# --- calibration_params: natural key = city, model, lead, month --------------------
calibration_params = sa.Table(
    "calibration_params",
    metadata,
    _id_column(),
    sa.Column("city", sa.Text, nullable=False),
    sa.Column("model", sa.Text, nullable=False),
    sa.Column("lead", sa.Integer, nullable=False),
    sa.Column("month", sa.Integer, nullable=False),
    _available_at_column(),
    sa.Index(
        "ix_calibration_params_latest",
        "city",
        "model",
        "lead",
        "month",
        "available_at",
    ),
)

# --- market_snapshots: natural key = ticker, snapshot_for (market time key) --------
market_snapshots = sa.Table(
    "market_snapshots",
    metadata,
    _id_column(),
    sa.Column("ticker", sa.Text, nullable=False),
    sa.Column("snapshot_for", sa.TIMESTAMP(timezone=True), nullable=False),
    _available_at_column(),
    sa.Index(
        "ix_market_snapshots_latest",
        "ticker",
        "snapshot_for",
        "available_at",
    ),
)

# --- fills: natural key = ticker, trade_id (order/leg identity) --------------------
fills = sa.Table(
    "fills",
    metadata,
    _id_column(),
    sa.Column("ticker", sa.Text, nullable=False),
    sa.Column("trade_id", sa.Text, nullable=False),
    _available_at_column(),
    sa.Index(
        "ix_fills_latest",
        "ticker",
        "trade_id",
        "available_at",
    ),
)


# --- Append-only enforcement (D-10) -------------------------------------------------
# Attach a per-table PostgreSQL rule that turns any UPDATE or DELETE into an error.
# Wiring it to the table's after_create / before_drop DDL events means the guard ships
# with BOTH metadata.create_all (used by the test fixture) and the Alembic migration —
# the schema is insert-only no matter how it is built. The ledger is corrected by
# INSERTing a new row with a later available_at, never by mutating an existing row.
for _table in metadata.tables.values():
    _name = _table.name
    _create_guard = sa.DDL(
        f'CREATE RULE "{_name}_no_update" AS ON UPDATE TO "{_name}" '
        f"DO INSTEAD SELECT raise_append_only('{_name}');\n"
        f'CREATE RULE "{_name}_no_delete" AS ON DELETE TO "{_name}" '
        f"DO INSTEAD SELECT raise_append_only('{_name}');"
    )
    _drop_guard = sa.DDL(
        f'DROP RULE IF EXISTS "{_name}_no_update" ON "{_name}";\n'
        f'DROP RULE IF EXISTS "{_name}_no_delete" ON "{_name}";'
    )
    sa.event.listen(_table, "after_create", _create_guard)
    sa.event.listen(_table, "before_drop", _drop_guard)


# A shared PL/pgSQL helper raises so any UPDATE/DELETE attempt fails loudly with a clear
# append-only message. Created before the first table, dropped after the last.
_create_raise_fn = sa.DDL(
    "CREATE OR REPLACE FUNCTION raise_append_only(tbl text) RETURNS void AS $$ "
    "BEGIN RAISE EXCEPTION 'append-only ledger: UPDATE/DELETE on % is forbidden "
    "(D-10) — correct via a new INSERT with a later available_at', tbl; END; "
    "$$ LANGUAGE plpgsql;"
)
_drop_raise_fn = sa.DDL("DROP FUNCTION IF EXISTS raise_append_only(text);")

sa.event.listen(metadata, "before_create", _create_raise_fn)
sa.event.listen(metadata, "after_drop", _drop_raise_fn)


__all__ = [
    "metadata",
    "forecasts",
    "observations",
    "calibration_params",
    "market_snapshots",
    "fills",
]
