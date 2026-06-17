"""CLI contract (02-05 Task 2/3) — argparse maps to the orchestrator backfill path (D-15).

The CLI is a THIN wrapper, so these tests pin its CONTRACT, not the orchestrator internals:

* a valid ``ingest`` invocation dispatches to ``orchestrator.ingest_range`` with the right
  models/cities/range and ``mode="backfill"`` (D-15);
* an unknown ``--city`` is rejected by argparse via ``get_city`` (ASVS V5 — a clear error);
* a malformed ``--date`` is rejected BEFORE any ingest call;
* a ``--start/--end`` range is forwarded as the inclusive range;
* ``--all-models`` / ``--all-cities`` expand to every model/city.

The orchestrator (and the DB engine) are MOCKED — no network, no Postgres. The scheduler
section (Task 3) asserts ``build_scheduler`` registers the per-model cadence jobs.
"""

from __future__ import annotations

from datetime import date

import pytest

from weatherquant import cli
from weatherquant.registry import CITIES


@pytest.fixture
def captured_range(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch the engine + async ingest_range so run_ingest is exercised offline.

    Records the args ``orchestrator.ingest_range`` was called with so the tests can assert the
    CLI mapped them correctly, and returns a deterministic per-model count.
    """
    captured: dict = {}

    def _fake_get_engine():
        return object()  # a sentinel bind — never touched (ingest_range is mocked).

    async def _fake_ingest_range(
        bind, models, cities, start_date, end_date, *, mode, lead, cycle_hours
    ):  # noqa: ANN001
        captured.update(
            bind=bind,
            models=list(models),
            cities=list(cities),
            start_date=start_date,
            end_date=end_date,
            mode=mode,
            lead=lead,
            cycle_hours=cycle_hours,
        )
        return {m: 1 for m in models}

    monkeypatch.setattr(cli, "get_engine", _fake_get_engine)
    monkeypatch.setattr(cli.orchestrator, "ingest_range", _fake_ingest_range)
    return captured


def test_single_date_single_city_single_model_dispatches_backfill(captured_range: dict):
    rc = cli.main(["ingest", "--model", "hrrr", "--city", "NYC", "--date", "2026-06-12"])
    assert rc == 0
    assert captured_range["models"] == ["hrrr"]
    assert captured_range["cities"] == ["NYC"]
    # A single --date collapses to a one-day inclusive range.
    assert captured_range["start_date"] == date(2026, 6, 12)
    assert captured_range["end_date"] == date(2026, 6, 12)
    # D-15: the CLI is the BACKFILL half of the one code path.
    assert captured_range["mode"] == "backfill"


def test_start_end_range_is_forwarded_inclusive(captured_range: dict):
    cli.main(
        [
            "ingest",
            "--model", "gfs",
            "--city", "CHI",
            "--start", "2026-06-10",
            "--end", "2026-06-12",
        ]
    )
    assert captured_range["start_date"] == date(2026, 6, 10)
    assert captured_range["end_date"] == date(2026, 6, 12)
    assert captured_range["mode"] == "backfill"


def test_all_models_and_all_cities_expand(captured_range: dict):
    cli.main(["ingest", "--all-models", "--all-cities", "--date", "2026-06-12"])
    assert set(captured_range["models"]) == set(cli.ALL_MODELS)
    assert set(captured_range["cities"]) == set(CITIES)


def test_unknown_city_is_rejected_before_any_ingest(
    captured_range: dict, capsys: pytest.CaptureFixture
):
    """An unknown --city is rejected by argparse (get_city) — ASVS V5, no ingest call."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["ingest", "--model", "hrrr", "--city", "ZZZ", "--date", "2026-06-12"])
    assert excinfo.value.code != 0  # argparse exits non-zero on a bad arg
    err = capsys.readouterr().err
    assert "ZZZ" in err  # the clear error names the bad code
    assert captured_range == {}  # the orchestrator was NEVER called


def test_malformed_date_is_rejected_before_any_ingest(
    captured_range: dict, capsys: pytest.CaptureFixture
):
    """A malformed --date is rejected before any ingest call (ASVS V5)."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["ingest", "--model", "hrrr", "--city", "NYC", "--date", "2026-13-99"])
    assert excinfo.value.code != 0
    err = capsys.readouterr().err
    assert "2026-13-99" in err
    assert captured_range == {}  # never dispatched


def test_cycle_hours_parsed_to_ints(captured_range: dict):
    cli.main(
        [
            "ingest",
            "--model", "gfs",
            "--city", "NYC",
            "--date", "2026-06-12",
            "--cycle-hours", "0,6,12,18",
        ]
    )
    assert captured_range["cycle_hours"] == [0, 6, 12, 18]


# --- Scheduler (02-05 Task 3) ------------------------------------------------------------


def test_build_scheduler_registers_per_model_jobs():
    """build_scheduler wires AsyncIOScheduler per model cadence WITHOUT starting (D-15)."""
    from weatherquant.scheduler import build_scheduler

    scheduler = build_scheduler()
    jobs = scheduler.get_jobs()
    # >=4 jobs registered (HRRR/NBM hourly + GFS/GEFS 00/06/12/18Z, plus obs/AFD cadence).
    assert len(jobs) >= 4
    # The scheduler is configured but NOT started (unit-testable).
    assert scheduler.running is False


def test_scheduler_is_asyncio_3x_not_4x():
    """The scheduler uses the apscheduler 3.11.x AsyncIOScheduler, not the 4.x API."""
    from apscheduler.schedulers.asyncio import AsyncIOScheduler

    from weatherquant.scheduler import build_scheduler

    assert isinstance(build_scheduler(), AsyncIOScheduler)


# --- price subcommand (04-05 Task 2) — parse-level contract, no DB needed -----------------


def test_price_subcommand_parses_valid_city_and_date():
    """A valid `price` invocation parses to the price command with city/date/lead/mid.

    This is the parse-level contract only (the orchestration in run_price reads the DB and is
    exercised offline elsewhere) — argparse validation runs BEFORE any DB call (ASVS V5).
    """
    parser = cli.build_parser()
    args = parser.parse_args(
        ["price", "--city", "NYC", "--date", "2026-06-12", "--ticker", "KXHIGHNY-62-63"]
    )
    assert args.command == "price"
    assert args.city == "NYC"
    assert args.date == date(2026, 6, 12)
    assert args.lead == 0  # default
    assert args.market_mid == 0.5  # mocked midpoint default (D-16 — no market fetch)
    assert args.ticker == "KXHIGHNY-62-63"


def test_price_unknown_city_is_rejected_by_argparse(capsys: pytest.CaptureFixture):
    """An unknown `price --city` is rejected by argparse via _city_type — ASVS V5 / T-04-15.

    The rejection happens at the arg edge, BEFORE any DB read in run_price.
    """
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["price", "--city", "ZZZ", "--date", "2026-06-12"])
    assert excinfo.value.code != 0  # argparse exits non-zero on a bad arg
    err = capsys.readouterr().err
    assert "ZZZ" in err  # the clear error names the bad code
