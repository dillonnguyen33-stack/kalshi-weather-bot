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

from datetime import datetime

import sqlalchemy as sa

from weatherquant.db import ddl

metadata = sa.MetaData()


# --- Sane INSERT rowcount (D-11 contract) ------------------------------------------
# Each ledger table has an ``Identity()`` surrogate PK, so SQLAlchemy appends an implicit
# ``RETURNING id`` to every INSERT. For a RETURNING insert the DBAPI cursor reports
# ``rowcount == -1`` (PEP 249 allows this) unless ``preserve_rowcount`` is set, which
# forces ``cursor.rowcount`` to be captured (yielding 1 for a single-row insert) without
# changing the emitted SQL. Callers across phases 2–6 rely on ``result.rowcount`` to
# confirm a write landed. This option is now set DETERMINISTICALLY on the engine built in
# :func:`weatherquant.db.engine.get_engine` (not as a fragile import-time global listener
# on the Engine class, which only worked by import-order luck) — see that module.


def _id_column() -> sa.Column[int]:
    """Surrogate serial primary key shared by every ledger table."""
    return sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True)


def _available_at_column() -> sa.Column[datetime]:
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
# ``snapshot_for`` is the market time-bucket key. It is typed ``Text`` (a stable string
# key, e.g. an ISO instant or market period label) rather than ``timestamptz`` so the
# natural key stays a simple, comparable identifier; the point-in-time-of-knowledge axis
# is ``available_at`` (D-12). Phase-5/6 may refine the encoding via a migration.
market_snapshots = sa.Table(
    "market_snapshots",
    metadata,
    _id_column(),
    sa.Column("ticker", sa.Text, nullable=False),
    sa.Column("snapshot_for", sa.Text, nullable=False),
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
# The enforcement DDL (the shared raise_append_only() function + per-table triggers) is
# single-sourced in weatherquant.db.ddl and consumed identically here (via create_all
# DDL events) and by the Alembic migration, so the two schemas cannot drift. Wiring the
# trigger DDL to each table's after_create / before_drop event ships the guard with BOTH
# metadata.create_all (test fixture) and the migration. The shared raise function is
# created before the first table and dropped after the last. Corrections are new INSERTs
# with a later available_at, never UPDATE/DELETE — see ddl.py for the full rationale
# (BEFORE-trigger choice, static RAISE message, % escaping trap).
for _table in metadata.tables.values():
    for _create_stmt in ddl.create_trigger_sql(_table.name):
        sa.event.listen(_table, "after_create", sa.DDL(_create_stmt))
    for _drop_stmt in ddl.drop_trigger_sql(_table.name):
        sa.event.listen(_table, "before_drop", sa.DDL(_drop_stmt))

sa.event.listen(metadata, "before_create", sa.DDL(ddl.CREATE_RAISE_FUNCTION_SQL))
sa.event.listen(metadata, "after_drop", sa.DDL(ddl.DROP_RAISE_FUNCTION_SQL))


__all__ = [
    "metadata",
    "forecasts",
    "observations",
    "calibration_params",
    "market_snapshots",
    "fills",
]
