"""SQLAlchemy Core schema for the append-only weatherquant ledger (D-10, D-11, D-12).

One :data:`metadata` object and the FIVE ledger tables every later phase writes to:

    forecasts · observations · calibration_params · market_snapshots · fills

Shared contract (D-11):

* **id** — surrogate serial PK (``BigInteger, Identity()``), NOT the natural key.
* **natural key** — business-identity columns; ``nullable=False`` and indexed
  (``ix_<table>_latest``) but NEVER UNIQUE (append-only: many rows share a key).
* **available_at** — ``TIMESTAMP(timezone=True), nullable=False``; point-in-time-of-
  knowledge (D-12), never back-dated (would break Phase 6's no-look-ahead walk-forward).

Insert-only — no UPDATE/DELETE (D-10); corrections are new inserts with a later
``available_at``, enforced structurally via DDL events (see :mod:`weatherquant.db.ddl`).
The "latest" read idiom lives in :mod:`weatherquant.db.queries`. SQLAlchemy Core only.
"""

from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from weatherquant.db import ddl

metadata = sa.MetaData()


# --- Sane INSERT rowcount (D-11 contract) ------------------------------------------
# The ``Identity()`` PK makes SQLAlchemy append an implicit ``RETURNING id``, so the
# cursor reports ``rowcount == -1`` unless ``preserve_rowcount`` is set. That option is
# set on the engine in :func:`weatherquant.db.engine.get_engine` — see that module.


def _id_column() -> sa.Column[int]:
    """Surrogate serial primary key shared by every ledger table."""
    return sa.Column("id", sa.BigInteger, sa.Identity(), primary_key=True)


def _available_at_column() -> sa.Column[datetime]:
    """Point-in-time-of-knowledge timestamptz shared by every ledger table (D-12)."""
    return sa.Column("available_at", sa.TIMESTAMP(timezone=True), nullable=False)


# --- forecasts: natural key = city, target_date, model, lead, member ---------------
# Phase-2 (02-01, D-05) extends the key with ``member`` (NOT NULL default 0 — deterministic
# models resolve to control; GEFS/Open-Meteo write 1..N) and adds GRIB payload columns.
# ``temp_kelvin`` stays Kelvin (D-07); ``cycle`` is model init time;
# ``station_lat``/``station_lon``/``grid_distance_m`` log the grid-point snap (D-05).
forecasts = sa.Table(
    "forecasts",
    metadata,
    _id_column(),
    sa.Column("city", sa.Text, nullable=False),
    sa.Column("target_date", sa.Date, nullable=False),
    sa.Column("model", sa.Text, nullable=False),
    sa.Column("lead", sa.Integer, nullable=False),
    sa.Column("member", sa.SmallInteger, nullable=False, server_default=sa.text("0")),
    sa.Column("temp_kelvin", sa.Float),
    sa.Column("cycle", sa.TIMESTAMP(timezone=True)),
    sa.Column("station_lat", sa.Float),
    sa.Column("station_lon", sa.Float),
    sa.Column("grid_distance_m", sa.Float),
    _available_at_column(),
    sa.Index(
        "ix_forecasts_latest",
        "city",
        "target_date",
        "model",
        "lead",
        "member",
        "available_at",
    ),
)

# --- observations: natural key = city, target_date (LST settlement day), source ----
# Phase-2 (02-01, D-06/D-07) adds obs payload columns; the key is UNCHANGED. ``daily_high_f``
# stays °F (D-07; conversion lives in ingest/obs.py). ``window_start``/``window_end`` are the
# LST settlement window; ``obs_count`` the in-window reading count; ``detail`` (JSONB) the
# AFD tool-use result / raw payload (D-06).
observations = sa.Table(
    "observations",
    metadata,
    _id_column(),
    sa.Column("city", sa.Text, nullable=False),
    sa.Column("target_date", sa.Date, nullable=False),
    sa.Column("source", sa.Text, nullable=False),
    sa.Column("daily_high_f", sa.Float),
    sa.Column("window_start", sa.TIMESTAMP(timezone=True)),
    sa.Column("window_end", sa.TIMESTAMP(timezone=True)),
    sa.Column("obs_count", sa.Integer),
    sa.Column("detail", postgresql.JSONB),
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
# Phase-3 (03-02, D-13) adds EMOS/NGR payload columns, all °F space (D-03), all nullable;
# the key is UNCHANGED (no member axis — members collapse into the mean/spread predictor,
# D-07). Params: ``mean_intercept``/``mean_slope`` (a, b: μ = a + b·m),
# ``var_intercept``/``var_slope`` (c, d: σ² = max(σ_floor², c² + d²·s²)); ``sigma_floor``
# the °F σ-clamp (D-09); ``n_train`` the sample count; ``pool_level`` the pooling-fallback
# provenance (D-08); ``crps_*`` the audit metrics (D-11); ``trained_through`` the data
# cutoff (D-13). Types mirror the 0003 migration EXACTLY (create_all == migrated schema).
calibration_params = sa.Table(
    "calibration_params",
    metadata,
    _id_column(),
    sa.Column("city", sa.Text, nullable=False),
    sa.Column("model", sa.Text, nullable=False),
    sa.Column("lead", sa.Integer, nullable=False),
    sa.Column("month", sa.Integer, nullable=False),
    sa.Column("mean_intercept", sa.Float, nullable=True),
    sa.Column("mean_slope", sa.Float, nullable=True),
    sa.Column("var_intercept", sa.Float, nullable=True),
    sa.Column("var_slope", sa.Float, nullable=True),
    sa.Column("sigma_floor", sa.Float, nullable=True),
    sa.Column("n_train", sa.Integer, nullable=True),
    sa.Column("pool_level", sa.Text, nullable=True),
    sa.Column("crps_train", sa.Float, nullable=True),
    sa.Column("crps_oos", sa.Float, nullable=True),
    sa.Column("crps_baseline_oos", sa.Float, nullable=True),
    sa.Column("trained_through", sa.Date, nullable=True),
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
# ``snapshot_for`` is the market time-bucket key, typed ``Text`` (stable comparable string,
# not timestamptz); ``available_at`` is the point-in-time axis (D-12).
#
# Phase-5 (05-00, D-11) adds Kalshi orderbook payload columns, all nullable.
# ``best_yes_bid``/``best_no_bid`` are top-of-book bids in integer cents (ask reflected as
# ``100 - opposite bid``, reflect.py, PAP-02); ``mid`` the derived midpoint fed to the
# Phase-4 EV/Kelly path (D-08/D-16); ``seq`` the WS orderbook_delta sequence (a gap triggers
# resnapshot, PAP-01); ``detail`` (JSONB) the raw book payload. Types must equal the applied
# migration (the migration test, not this comment, enforces parity).
#
# MONEY-UNIT LANDMARK: ``mid`` is FLOAT-VALUED CENTS (half-cent midpoint, unit-consistent with
# best_*_bid and the fill's avg_price_cents) — NOT [0,1] dollars. MUST stay ``sa.Float``:
# rounding to int would inject +/-0.5c bias into every derived CLV. ``volume`` is the
# supporting book-liquidity in WHOLE CONTRACTS — ``min(best_yes_bid_size, best_no_bid_size)``,
# the size behind THIS mid (not two-sided union depth) — weights the CLV closing mid.
market_snapshots = sa.Table(
    "market_snapshots",
    metadata,
    _id_column(),
    sa.Column("ticker", sa.Text, nullable=False),
    sa.Column("snapshot_for", sa.Text, nullable=False),
    sa.Column("best_yes_bid", sa.Integer, nullable=True),
    sa.Column("best_no_bid", sa.Integer, nullable=True),
    # mid: FLOAT-VALUED CENTS (not [0,1] dollars); stays Float so no +/-0.5c rounding bias.
    sa.Column("mid", sa.Float, nullable=True),
    # volume: supporting book-liquidity in WHOLE CONTRACTS, min(best_yes_bid_size,
    # best_no_bid_size); weights the CLV closing mid.
    sa.Column("volume", sa.Integer, nullable=True),
    sa.Column("seq", sa.BigInteger, nullable=True),
    sa.Column("detail", postgresql.JSONB, nullable=True),
    _available_at_column(),
    sa.Index(
        "ix_market_snapshots_latest",
        "ticker",
        "snapshot_for",
        "available_at",
    ),
)

# --- fills: natural key = ticker, trade_id (order/leg identity) --------------------
# Phase-5 (05-00, D-11) adds simulated-fill payload columns, all nullable. Execution: ``side``
# (yes/no), ``price`` (Integer cents), ``count`` (contracts), ``fee`` (Integer cents, exact_fee),
# ``is_maker`` (Boolean, PAP-02/PAP-03), ``event_time`` (the REAL WS event time, never back-dated,
# D-08/PAP-03). Intent linkage to the Phase-4 money path: ``bucket_prob``/``ev``/``kelly_stake``
# (Float) make each fill auditable against its forecast. ``detail`` (JSONB) the raw trade payload;
# key UNCHANGED. Types mirror 0004 EXACTLY (create_all == migrated).
#
# PRECISION CONTRACT (WR-05): ``price`` is the ROUNDED whole-cent fill price — for the 1..99c band
# guard and display only, NOT money math (it carries the same +/-0.5c bias the float ``mid``
# avoids). Derived CLV MUST read ``detail['avg_price_cents']`` (float cents), never the integer
# ``price`` column; ``clv.clv_cents`` already consumes that float.
fills = sa.Table(
    "fills",
    metadata,
    _id_column(),
    sa.Column("ticker", sa.Text, nullable=False),
    sa.Column("trade_id", sa.Text, nullable=False),
    sa.Column("side", sa.Text, nullable=True),
    sa.Column("price", sa.Integer, nullable=True),
    sa.Column("count", sa.Integer, nullable=True),
    sa.Column("fee", sa.Integer, nullable=True),
    sa.Column("is_maker", sa.Boolean, nullable=True),
    sa.Column("event_time", sa.TIMESTAMP(timezone=True), nullable=True),
    sa.Column("bucket_prob", sa.Float, nullable=True),
    sa.Column("ev", sa.Float, nullable=True),
    sa.Column("kelly_stake", sa.Float, nullable=True),
    sa.Column("detail", postgresql.JSONB, nullable=True),
    _available_at_column(),
    sa.Index(
        "ix_fills_latest",
        "ticker",
        "trade_id",
        "available_at",
    ),
)


# --- Natural keys: the single source of truth for row identity (D-10/D-12) ---------
# The COMPLETE key per table — ``latest`` MUST DISTINCT ON it all, or it collapses distinct
# facts (e.g. ensemble members). Mirrors each ``ix_<table>_latest`` minus ``available_at``.
NATURAL_KEYS: dict[str, tuple[str, ...]] = {
    "forecasts": ("city", "target_date", "model", "lead", "member"),
    "observations": ("city", "target_date", "source"),
    "calibration_params": ("city", "model", "lead", "month"),
    "market_snapshots": ("ticker", "snapshot_for"),
    "fills": ("ticker", "trade_id"),
}


# --- Append-only enforcement (D-10) -------------------------------------------------
# DDL single-sourced in weatherquant.db.ddl; wired to each table's after_create/before_drop
# (and the metadata before_create/after_drop for the shared function) so the guard ships with
# both create_all and the migration. See ddl.py for the full rationale.


def _ddl(stmt: str) -> sa.DDL:
    # sa.DDL lacks a typed stub; one wrapper confines the mypy no-untyped-call suppression.
    return sa.DDL(stmt)  # type: ignore[no-untyped-call]


for _table in metadata.tables.values():
    for _create_stmt in ddl.create_trigger_sql(_table.name):
        sa.event.listen(_table, "after_create", _ddl(_create_stmt))
    for _drop_stmt in ddl.drop_trigger_sql(_table.name):
        sa.event.listen(_table, "before_drop", _ddl(_drop_stmt))

sa.event.listen(metadata, "before_create", _ddl(ddl.CREATE_RAISE_FUNCTION_SQL))
sa.event.listen(metadata, "after_drop", _ddl(ddl.DROP_RAISE_FUNCTION_SQL))


__all__ = [
    "NATURAL_KEYS",
    "calibration_params",
    "fills",
    "forecasts",
    "market_snapshots",
    "metadata",
    "observations",
]
