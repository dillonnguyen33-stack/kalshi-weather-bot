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
from sqlalchemy.dialects import postgresql

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


# --- forecasts: natural key = city, target_date, model, lead, member ---------------
# Phase-2 (02-01, D-05) extends the natural key with ``member`` (ensemble member axis)
# and adds the decoded GRIB payload columns. ``member`` is NOT NULL with a server default
# of 0 so the deterministic models (HRRR/GFS/NBM) and any pre-existing rows resolve to the
# control member; GEFS/Open-Meteo members write 1..N. ``temp_kelvin`` keeps forecasts in
# Kelvin (units never leak to °F on the forecast path — D-07); ``cycle`` is the model
# init time; ``station_lat``/``station_lon``/``grid_distance_m`` log the nearest-grid-point
# snap to the Kalshi settlement station (D-05). Columns are DOUBLE PRECISION (sa.Float).
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
# Phase-2 (02-01, D-06/D-07) adds the obs payload columns. The natural key
# (city, target_date, source) is UNCHANGED — ``source='afd'`` slots in alongside the
# weather-feed sources. ``daily_high_f`` keeps observations in °F (D-07; the °F→K
# conversion is centralized in ingest/obs.py, NOT here). ``window_start``/``window_end``
# record the LST settlement window the daily high was computed over; ``obs_count`` is the
# number of in-window readings; ``detail`` (JSONB) carries the AFD tool-use result / raw
# obs payload (D-06).
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
# Phase-3 (03-02, D-13) extends this table with the EMOS/NGR payload columns, all in °F
# space (D-03) and all ``nullable=True`` so the column-add is non-breaking. The natural
# key (city, model, lead, month) is UNCHANGED — members collapse into the (mean, spread)
# predictor, so calibration carries no member axis (D-07). The interpretable params are
# ``mean_intercept``/``mean_slope`` (a, b: μ = a + b·m) and ``var_intercept``/``var_slope``
# (c, d: σ² = max(σ_floor², c² + d²·s²)); ``sigma_floor`` is the °F σ-clamp (D-09);
# ``n_train`` the samples used; ``pool_level`` the pooling-fallback provenance (D-08);
# ``crps_train``/``crps_oos``/``crps_baseline_oos`` the audit metrics (D-11); and
# ``trained_through`` the data cutoff so Phase 6 can re-derive any historical fit (D-13).
# Types mirror the 0003 migration EXACTLY so metadata.create_all == the migrated schema.
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
# ``snapshot_for`` is the market time-bucket key. It is typed ``Text`` (a stable string
# key, e.g. an ISO instant or market period label) rather than ``timestamptz`` so the
# natural key stays a simple, comparable identifier; the point-in-time-of-knowledge axis
# is ``available_at`` (D-12).
#
# Phase-5 (05-00, D-11) adds the Kalshi orderbook payload columns — all ``nullable=True`` so
# the column-add is non-breaking. ``best_yes_bid``/``best_no_bid`` are the top-of-book bids in
# integer cents (the only side Kalshi quotes — the ask side is reflected as ``100 - opposite
# bid`` in market/reflect.py, PAP-02); ``mid`` is the derived midpoint (Float, the real market
# midpoint fed into the Phase-4 EV/Kelly path, closing the D-08/D-16 loop); ``seq`` is the WS
# orderbook_delta sequence number (BigInteger — a gap triggers a resnapshot, PAP-01); and
# ``detail`` (JSONB) carries the raw book payload, mirroring ``observations.detail``. The
# natural key is (ticker, snapshot_for) so ix_market_snapshots_latest keys the latest snapshot.
# Column types must equal the applied migration so metadata.create_all reproduces the migrated
# schema (the migration test, not this comment, enforces that parity).
#
# MONEY-UNIT LANDMARK: ``mid`` is FLOAT-VALUED CENTS — the half-cent bid/ask midpoint,
# unit-consistent with ``best_yes_bid``/``best_no_bid`` (integer cents) and with the fill's
# ``avg_price_cents`` (cents). It is NOT [0,1] dollars. Keeping one unit lets
# ``clv.vol_weighted_mid``/``clv_cents`` subtract ``avg_price_cents`` directly with no
# conversion (``run_paper`` persists the un-divided ``mid_cents`` here and keeps the [0,1]
# ``mid_unit`` only for the p_used/EV/Kelly pricing math). ``mid`` MUST stay ``sa.Float`` (NOT
# Integer) — rounding to an int would inject +/-0.5c bias into every derived CLV. ``volume`` is
# the per-snapshot book-liquidity signal in WHOLE CONTRACTS (Integer) consumed by
# ``vol_weighted_mid`` to weight the closing mid: it is the top-of-book size SUPPORTING the
# persisted yes mid — ``min(best_yes_bid_size, best_no_bid_size)``, the liquidity BEHIND THIS
# mid (the best-no-bid size IS the reflected best-yes-ask size, reflect.py). It is the
# supporting size, NOT the two-sided UNION depth, which would over-weight an opposite-side-deep
# but thinly-supported mid while still being a REAL feed signal (not a fixture).
market_snapshots = sa.Table(
    "market_snapshots",
    metadata,
    _id_column(),
    sa.Column("ticker", sa.Text, nullable=False),
    sa.Column("snapshot_for", sa.Text, nullable=False),
    sa.Column("best_yes_bid", sa.Integer, nullable=True),
    sa.Column("best_no_bid", sa.Integer, nullable=True),
    # mid is FLOAT-VALUED CENTS (the half-cent midpoint, unit-consistent with best_*_bid and
    # avg_price_cents) — NOT [0,1] dollars; stays Float so no +/-0.5c rounding bias.
    sa.Column("mid", sa.Float, nullable=True),
    # volume: per-snapshot book-liquidity signal in WHOLE CONTRACTS — the top-of-book size
    # SUPPORTING the persisted yes mid (min(best_yes_bid_size, best_no_bid_size)), the liquidity
    # behind THIS mid; weights the CLV closing mid.
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
# Phase-5 (05-00, D-11) adds the simulated-fill payload columns — all ``nullable=True``. The
# execution payload: ``side`` (Text, yes/no), ``price`` (Integer cents), ``count`` (Integer
# contracts), ``fee`` (Integer cents — the exact_fee per order), ``is_maker`` (Boolean —
# maker queue vs taker sweep, PAP-02/PAP-03), and ``event_time`` (timestamptz — the REAL WS
# event time the fill occurred at, never now()/back-dated, D-08, PAP-03). The intent linkage
# back to the Phase-4 money path: ``bucket_prob``/``ev``/``kelly_stake`` (Float) record the
# model probability, expected value, and Kelly stake that motivated the order, so each fill is
# auditable against the forecast that produced it. ``detail`` (JSONB) carries the raw trade
# payload, mirroring ``observations.detail``. The natural key (ticker, trade_id) is UNCHANGED
# so ix_fills_latest is intact. Types mirror 0004 EXACTLY (metadata.create_all == migrated).
#
# PRECISION CONTRACT (WR-05): ``price`` is the ROUNDED whole-cent fill price — it exists for the
# 1..99c band guard and human-readable display, NOT for derived money math. It carries a
# +/-0.5c rounding bias, exactly the bias the ``market_snapshots.mid`` column was kept ``Float``
# to avoid. Derived CLV MUST read the un-rounded size-weighted average from
# ``detail['avg_price_cents']`` (float cents), never the integer ``price`` column — reconstructing
# CLV from ``price`` would re-introduce the +/-0.5c bias the float ``mid`` deliberately avoids.
# ``clv.clv_cents`` already consumes the float ``avg_price_cents`` off the Fill object / detail.
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
# The natural key is the business identity of a fact; the "latest" read idiom
# (weatherquant.db.queries.latest) MUST DISTINCT ON the COMPLETE key, or it silently
# collapses genuinely-distinct facts (e.g. two ensemble members) into one row. Keyed by
# table so every phase-2+ call site looks the key up instead of passing it by hand and
# risking an under-specified key. Mirrors each ``ix_<table>_latest`` index minus the
# trailing ``available_at`` (the point-in-time axis, not part of identity).
NATURAL_KEYS: dict[str, tuple[str, ...]] = {
    "forecasts": ("city", "target_date", "model", "lead", "member"),
    "observations": ("city", "target_date", "source"),
    "calibration_params": ("city", "model", "lead", "month"),
    "market_snapshots": ("ticker", "snapshot_for"),
    "fills": ("ticker", "trade_id"),
}


# --- Append-only enforcement (D-10) -------------------------------------------------
# The enforcement DDL (the shared raise_append_only() function + per-table triggers) is
# single-sourced in weatherquant.db.ddl and consumed identically here (via create_all
# DDL events) and by the Alembic migration, so the two schemas cannot drift. Wiring the
# trigger DDL to each table's after_create / before_drop event ships the guard with BOTH
# metadata.create_all (test fixture) and the migration. The shared raise function is
# created before the first table and dropped after the last. Corrections are new INSERTs
# with a later available_at, never UPDATE/DELETE — see ddl.py for the full rationale
# (BEFORE-trigger choice, static RAISE message, % escaping trap).


def _ddl(stmt: str) -> sa.DDL:
    # sa.DDL lacks a typed stub, so a bare call trips mypy --strict's no-untyped-call.
    # One wrapper confines that single suppression instead of repeating it per call site.
    return sa.DDL(stmt)  # type: ignore[no-untyped-call]


for _table in metadata.tables.values():
    for _create_stmt in ddl.create_trigger_sql(_table.name):
        sa.event.listen(_table, "after_create", _ddl(_create_stmt))
    for _drop_stmt in ddl.drop_trigger_sql(_table.name):
        sa.event.listen(_table, "before_drop", _ddl(_drop_stmt))

sa.event.listen(metadata, "before_create", _ddl(ddl.CREATE_RAISE_FUNCTION_SQL))
sa.event.listen(metadata, "after_drop", _ddl(ddl.DROP_RAISE_FUNCTION_SQL))


__all__ = [
    "metadata",
    "NATURAL_KEYS",
    "forecasts",
    "observations",
    "calibration_params",
    "market_snapshots",
    "fills",
]
