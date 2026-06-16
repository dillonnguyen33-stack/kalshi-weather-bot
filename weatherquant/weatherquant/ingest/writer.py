"""The SINGLE audited write path for both ledger payload tables (D-10 / D-11).

Every forecast AND every observation is inserted through THIS module — never a hand-rolled
Core insert in ``grib.py`` / ``obs.py`` / ``afd.py``. Two public entry points share one
private helper so there is exactly ONE place that:

1. runs the content/cycle skip-before-insert (``idempotency.row_exists``) FIRST, returning
   0 (a no-op skip) when an identical row already exists (D-10), and
2. otherwise executes a SQLAlchemy Core ``table.insert().values(...)`` (Core only, no ORM)
   and asserts ``result.rowcount == 1`` — the preserve_rowcount contract (D-11).

There is NO UPDATE / upsert / ON CONFLICT path anywhere here: the append-only trigger
(:mod:`weatherquant.db.ddl`) would raise, so a correction is a fresh INSERT with a later
``available_at`` (D-10). ``insert_forecast`` targets ``forecasts`` (Kelvin payload);
``insert_observation`` targets ``observations`` (°F payload + AFD ``detail`` jsonb). Both
expect a ``bind`` built by :func:`weatherquant.db.engine.get_engine` so ``preserve_rowcount``
holds (a single-row insert reports ``rowcount == 1`` despite the implicit ``RETURNING id``).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine

from weatherquant.db.models import forecasts, observations
from weatherquant.ingest.idempotency import row_exists


def _insert_row(
    bind: Engine | Connection,
    table: sa.Table,
    natural_key: Mapping[str, object],
    content: Mapping[str, object],
    available_at: datetime,
) -> int:
    """Skip-before-insert one ledger row; return rowcount (1 inserted, 0 skipped).

    The ONE audited insert path (D-10/D-11). Calls ``row_exists`` over the natural key +
    content FIRST; if the identical row is already present, returns 0 (no-op skip) without
    touching the table. Otherwise executes a Core insert and asserts ``rowcount == 1``.
    Never issues an UPDATE/upsert — the append-only trigger would raise.
    """

    def _do(conn: Connection) -> int:
        if row_exists(conn, table.name, natural_key, content):
            return 0  # identical row already in the ledger — skip (D-10), no UPDATE.
        values = {**natural_key, **content, "available_at": available_at}
        result = conn.execute(table.insert().values(**values))
        # preserve_rowcount (engine.get_engine) makes a single-row insert report 1 despite
        # the implicit RETURNING id on the Identity() PK (D-11 contract).
        assert result.rowcount == 1, (
            f"expected rowcount==1 inserting into {table.name}, got {result.rowcount}"
        )
        return result.rowcount

    if isinstance(bind, Engine):
        with bind.begin() as conn:
            return _do(conn)
    return _do(bind)


def insert_forecast(
    bind: Engine | Connection,
    *,
    city: str,
    target_date: date,
    model: str,
    lead: int,
    member: int,
    temp_kelvin: float,
    cycle: datetime,
    station_lat: float,
    station_lon: float,
    grid_distance_m: float,
    available_at: datetime,
) -> int:
    """Insert one forecast row through the single audited path (D-11).

    Natural key: ``(city, target_date, model, lead, member, cycle)``. Content:
    ``temp_kelvin`` + the station snap fields. Re-inserting an identical cycle is a no-op
    (returns 0); a changed ``temp_kelvin`` appends a fresh row (returns 1). Forecasts are
    Kelvin-only (D-07) — °F never enters this path.

    Returns:
        ``1`` if a row was inserted, ``0`` if an identical row already existed (skip).
    """
    natural_key = {
        "city": city,
        "target_date": target_date,
        "model": model,
        "lead": lead,
        "member": member,
        "cycle": cycle,
    }
    content = {
        "temp_kelvin": temp_kelvin,
        "station_lat": station_lat,
        "station_lon": station_lon,
        "grid_distance_m": grid_distance_m,
    }
    return _insert_row(bind, forecasts, natural_key, content, available_at)


def insert_observation(
    bind: Engine | Connection,
    *,
    city: str,
    target_date: date,
    source: str,
    daily_high_f: float | None = None,
    window_start: datetime | None = None,
    window_end: datetime | None = None,
    obs_count: int | None = None,
    detail: Mapping[str, object] | None = None,
    available_at: datetime,
) -> int:
    """Insert one observation row through the SAME audited path as forecasts (D-10/D-11).

    Natural key: ``(city, target_date, source)`` — ``source='afd'`` slots in alongside the
    weather-feed sources. Content: the °F daily-high payload (``daily_high_f`` /
    ``window_start`` / ``window_end`` / ``obs_count``) and the ``detail`` jsonb (AFD tool
    result / raw obs payload). 02-03's ``obs.py`` / ``afd.py`` MUST route through here
    rather than a hand-rolled Core insert, so there is exactly one idempotency + rowcount
    contract for observations too.

    Returns:
        ``1`` if a row was inserted, ``0`` if an identical row already existed (skip).
    """
    natural_key = {
        "city": city,
        "target_date": target_date,
        "source": source,
    }
    content = {
        "daily_high_f": daily_high_f,
        "window_start": window_start,
        "window_end": window_end,
        "obs_count": obs_count,
        "detail": detail,
    }
    return _insert_row(bind, observations, natural_key, content, available_at)


__all__ = ["insert_forecast", "insert_observation"]
