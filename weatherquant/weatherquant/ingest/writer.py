"""The SINGLE audited write path for both ledger payload tables (D-10 / D-11).

Every forecast AND every observation is inserted through THIS module — never a hand-rolled
Core insert in ``grib.py`` / ``obs.py`` / ``afd.py``. Two public entry points share one
private helper so there is exactly ONE place that:

1. runs the content/cycle skip-before-insert (``idempotency.row_exists``) FIRST, returning
   0 (a no-op skip) when an identical row already exists (D-10), and
2. otherwise executes a SQLAlchemy Core ``table.insert().values(...)`` (Core only, no ORM)
   and raises ``RuntimeError`` unless ``result.rowcount == 1`` — the preserve_rowcount
   contract (D-11). An explicit raise (not a bare ``assert``) so the guard survives
   ``python -O`` (WR-06).

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

from weatherquant.db.models import fills, forecasts, market_snapshots, observations
from weatherquant.db.types import Bind
from weatherquant.ingest.errors import CorrectnessError
from weatherquant.ingest.idempotency import row_exists


class WriteIntegrityError(CorrectnessError, RuntimeError):
    """The single-row insert integrity contract was violated (D-11, WR-06).

    Raised when the audited insert reports ``rowcount != 1``. A :class:`CorrectnessError`
    (and still a ``RuntimeError`` for back-compat) so the orchestrator's graceful-degradation
    handler (WR-05) lets this CORRECTNESS ALARM propagate loudly via the single
    ``except CorrectnessError`` re-raise — a row that should have landed silently vanishing is a
    real bug, not a backfillable gap.
    """


def _insert_row(
    bind: Bind,
    table: sa.Table,
    natural_key: Mapping[str, object],
    content: Mapping[str, object],
    available_at: datetime,
) -> int:
    """Skip-before-insert one ledger row; return rowcount (1 inserted, 0 skipped).

    The ONE audited insert path (D-10/D-11). Calls ``row_exists`` over the natural key +
    content FIRST; if the identical row is already present, returns 0 (no-op skip) without
    touching the table. Otherwise executes a Core insert and raises ``RuntimeError`` unless
    ``rowcount == 1`` (WR-06: an explicit raise, not a stripped-under-``-O`` assert).
    Never issues an UPDATE/upsert — the append-only trigger would raise.
    """

    def _do(conn: Connection) -> int:
        if row_exists(conn, table.name, natural_key, content):
            return 0  # identical row already in the ledger — skip (D-10), no UPDATE.
        values = {**natural_key, **content, "available_at": available_at}
        result = conn.execute(table.insert().values(**values))
        # Bind result.rowcount (typed ``Any`` by the SQLAlchemy stubs) to an int local on the
        # ONE audited insert path whose whole purpose is the rowcount==1 integrity contract
        # (TS-2, D-11): typing the guard's input means a future ``rowcount -> int | None``
        # widening regression is caught by mypy rather than silently slipping past the check.
        rowcount: int = result.rowcount
        # preserve_rowcount (engine.get_engine) makes a single-row insert report 1 despite
        # the implicit RETURNING id on the Identity() PK (D-11 contract). This integrity
        # guard is an explicit raise, NOT a bare assert: `python -O` / PYTHONOPTIMIZE strips
        # asserts, which would silently disable the only check that the single audited insert
        # actually landed a row (WR-06).
        if rowcount != 1:
            raise WriteIntegrityError(
                f"expected rowcount==1 inserting into {table.name}, got {rowcount}"
            )
        return rowcount

    if isinstance(bind, Engine):
        with bind.begin() as conn:
            return _do(conn)
    return _do(bind)


def insert_forecast(
    bind: Bind,
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
    bind: Bind,
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


def insert_market_snapshot(
    bind: Bind,
    *,
    ticker: str,
    snapshot_for: str,
    best_yes_bid: int | None = None,
    best_no_bid: int | None = None,
    mid: float | None = None,
    volume: int | None = None,
    seq: int | None = None,
    detail: Mapping[str, object] | None = None,
    available_at: datetime,
) -> int:
    """Insert one market-snapshot row through the SAME audited path (D-10/D-11/D-13).

    Natural key: ``(ticker, snapshot_for)`` — ``snapshot_for`` is the stable market
    time-bucket key (a string, e.g. an ISO instant; see ``db/models.py``). Content: the
    load-bearing top-of-book fields (``best_yes_bid``/``best_no_bid`` in integer cents, the
    only side Kalshi quotes — the ask is reflected as ``100 - opposite bid`` in
    ``market/reflect.py``), the derived ``mid`` (the real market midpoint in FLOAT-VALUED
    CENTS — unit-consistent with ``best_*_bid``/``avg_price_cents``, CR-01), ``volume`` (the
    per-snapshot book-liquidity volume signal in WHOLE CONTRACTS — the summed resting
    top-of-book size off the live orderbook payload; weights the CLV closing mid, WR-01), the
    WS ``seq``, and the raw book payload ``detail`` (JSONB, mirroring ``observations.detail``).
    Omitting ``volume`` persists NULL (back-compat). Re-inserting an identical snapshot is a no-op
    (returns 0); a changed payload appends a fresh row (returns 1). NO UPDATE/upsert — the
    append-only trigger would raise; a correction is a later-``available_at`` INSERT.

    ``available_at`` is ALWAYS a caller param — the REAL WS event time the snapshot was
    observed at (D-08), NEVER ``now()`` inside the writer.

    Returns:
        ``1`` if a row was inserted, ``0`` if an identical row already existed (skip).
    """
    natural_key = {
        "ticker": ticker,
        "snapshot_for": snapshot_for,
    }
    content = {
        "best_yes_bid": best_yes_bid,
        "best_no_bid": best_no_bid,
        "mid": mid,
        "volume": volume,
        "seq": seq,
        "detail": detail,
    }
    return _insert_row(bind, market_snapshots, natural_key, content, available_at)


def insert_fill(
    bind: Bind,
    *,
    ticker: str,
    trade_id: str,
    side: str | None = None,
    price: int | None = None,
    count: int | None = None,
    fee: int | None = None,
    is_maker: bool | None = None,
    event_time: datetime | None = None,
    bucket_prob: float | None = None,
    ev: float | None = None,
    kelly_stake: float | None = None,
    detail: Mapping[str, object] | None = None,
    available_at: datetime,
) -> int:
    """Insert one simulated-fill row through the SAME audited path (D-10/D-11/D-13).

    Natural key: ``(ticker, trade_id)``. Content: the execution payload (``side`` yes/no,
    ``price`` integer cents, ``count`` contracts, ``fee`` integer cents, ``is_maker`` maker
    vs taker, ``event_time`` the REAL WS event time the fill occurred at), the intent
    linkage back to the Phase-4 money path (``bucket_prob``/``ev``/``kelly_stake``), and the
    raw trade payload ``detail`` (JSONB). Re-inserting an identical fill is a no-op
    (returns 0); a changed payload appends a fresh row (returns 1). NO UPDATE/upsert.

    ``available_at`` is ALWAYS a caller param (the real WS event time, D-08), NEVER ``now()``
    inside the writer. ``event_time`` is the fill's own observed instant carried in the
    content payload (also never back-dated by the writer).

    Returns:
        ``1`` if a row was inserted, ``0`` if an identical row already existed (skip).
    """
    # A maker rests AT a known price, so a maker fill MUST carry a real resting price. fills.py's
    # maker_queue_fill returns avg_price_cents=0.0 as an OUT-OF-BAND placeholder (the queue model
    # proves the COUNT; taker is the credited Gate-1 path) — the caller is meant to supply the
    # resting price out of band. Persisting that un-supplied 0c through here would silently stamp
    # price=0 on a money field that feeds CLV as closing_mid - 0 (CORR-MED-4). Fail loud with the
    # established money-path correctness alarm (same WriteIntegrityError the rowcount==1 contract
    # raises, surviving python -O) so the contract violation surfaces at the audited write
    # boundary rather than corrupting CLV downstream. Taker fills (is_maker is not True) are
    # unaffected.
    if is_maker is True and (price is None or price == 0):
        raise WriteIntegrityError(
            f"refusing to persist a maker fill (ticker={ticker}, trade_id={trade_id}) with "
            f"price={price!r}: a maker rests at a real resting price; price in {{None, 0}} is "
            "the out-of-band maker_queue_fill placeholder and would corrupt CLV as "
            "closing_mid - 0 (CORR-MED-4)."
        )
    natural_key = {
        "ticker": ticker,
        "trade_id": trade_id,
    }
    content = {
        "side": side,
        "price": price,
        "count": count,
        "fee": fee,
        "is_maker": is_maker,
        "event_time": event_time,
        "bucket_prob": bucket_prob,
        "ev": ev,
        "kelly_stake": kelly_stake,
        "detail": detail,
    }
    return _insert_row(bind, fills, natural_key, content, available_at)


__all__ = [
    "insert_forecast",
    "insert_observation",
    "insert_market_snapshot",
    "insert_fill",
    "WriteIntegrityError",
]
