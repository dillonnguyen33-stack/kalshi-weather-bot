"""ING-08 / D-10/D-11 (integration): skip-before-insert idempotency + single write path.

GREEN by 02-02's ``weatherquant.ingest.idempotency`` + ``weatherquant.ingest.writer``.
Idempotency is a SKIP-before-insert (never an UPDATE/upsert — the append-only trigger
would raise, Pitfall 6). Marked ``integration``: skips cleanly when DATABASE_URL is unset.

Covers:
* re-ingesting an identical forecast cycle is a no-op (COUNT(*) and latest() unchanged);
* a changed payload inserts exactly one new row with a later available_at (latest() moves);
* insert_observation runs the SAME skip-before-insert (idempotent on repeated obs);
* neither path provokes the append-only trigger.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
import sqlalchemy as sa

from weatherquant.db.queries import latest
from weatherquant.ingest.idempotency import already_ingested, row_exists
from weatherquant.ingest.writer import insert_forecast, insert_observation

pytestmark = pytest.mark.integration

_FORECAST_KEY = ["city", "target_date", "model", "lead", "member"]
_OBS_KEY = ["city", "target_date", "source"]


def test_idempotency_predicate_is_callable():
    # The Wave-0 stub asserted this; already_ingested is the skip-before-insert predicate.
    assert callable(already_ingested)
    assert already_ingested is row_exists


def _count(engine, table_name: str) -> int:
    with engine.connect() as conn:
        return conn.execute(
            sa.text(f"SELECT COUNT(*) FROM {table_name}")  # noqa: S608 - test-only literal
        ).scalar_one()


def test_reingesting_identical_cycle_inserts_no_duplicate(pg_engine):
    cycle = datetime(2026, 6, 13, 12, 0, tzinfo=timezone.utc)
    avail = cycle + timedelta(hours=1, minutes=30)
    kwargs = dict(
        city="NYC",
        target_date=date(2026, 6, 13),
        model="hrrr",
        lead=0,
        member=0,
        temp_kelvin=300.15,
        cycle=cycle,
        station_lat=40.779,
        station_lon=-73.969,
        grid_distance_m=1234.0,
        available_at=avail,
    )

    before = _count(pg_engine, "forecasts")

    # First write inserts (rowcount 1).
    assert insert_forecast(pg_engine, **kwargs) == 1
    # Identical re-run is a no-op skip (rowcount 0) — no trigger, no duplicate.
    assert insert_forecast(pg_engine, **kwargs) == 0

    assert _count(pg_engine, "forecasts") == before + 1

    rows = [r for r in latest(pg_engine, "forecasts", _FORECAST_KEY) if r["city"] == "NYC"]
    nyc = [r for r in rows if r["target_date"] == date(2026, 6, 13) and r["model"] == "hrrr"]
    assert len(nyc) == 1
    assert nyc[0]["temp_kelvin"] == pytest.approx(300.15)


def test_changed_payload_appends_one_new_row_with_later_available_at(pg_engine):
    cycle = datetime(2026, 6, 14, 12, 0, tzinfo=timezone.utc)
    base = dict(
        city="CHI",
        target_date=date(2026, 6, 14),
        model="hrrr",
        lead=0,
        member=0,
        cycle=cycle,
        station_lat=41.786,
        station_lon=-87.752,
        grid_distance_m=900.0,
    )
    avail1 = cycle + timedelta(hours=1, minutes=30)
    avail2 = avail1 + timedelta(hours=1)

    assert insert_forecast(pg_engine, **base, temp_kelvin=295.0, available_at=avail1) == 1
    before = _count(pg_engine, "forecasts")
    # A corrected value (different temp_kelvin) does NOT match -> inserts a fresh row.
    assert insert_forecast(pg_engine, **base, temp_kelvin=296.5, available_at=avail2) == 1
    assert _count(pg_engine, "forecasts") == before + 1

    rows = latest(pg_engine, "forecasts", _FORECAST_KEY)
    chi = [
        r for r in rows
        if r["city"] == "CHI" and r["target_date"] == date(2026, 6, 14) and r["model"] == "hrrr"
    ]
    assert len(chi) == 1
    # latest() returns the corrected value (later available_at wins).
    assert chi[0]["temp_kelvin"] == pytest.approx(296.5)


def test_insert_observation_is_idempotent_same_path(pg_engine):
    target = date(2026, 6, 13)
    win_start = datetime(2026, 6, 13, 5, 0, tzinfo=timezone.utc)
    win_end = win_start + timedelta(days=1)
    kwargs = dict(
        city="NYC",
        target_date=target,
        source="asos",
        daily_high_f=88.0,
        window_start=win_start,
        window_end=win_end,
        obs_count=24,
        available_at=win_end,
    )

    before = _count(pg_engine, "observations")
    assert insert_observation(pg_engine, **kwargs) == 1
    # Same (city, target_date, source) + identical content -> skip (same audited path).
    assert insert_observation(pg_engine, **kwargs) == 0
    assert _count(pg_engine, "observations") == before + 1

    rows = latest(pg_engine, "observations", _OBS_KEY)
    nyc = [r for r in rows if r["city"] == "NYC" and r["source"] == "asos" and r["target_date"] == target]
    assert len(nyc) == 1
    assert nyc[0]["daily_high_f"] == pytest.approx(88.0)
