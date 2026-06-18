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


# --- run_price orchestration (WR-A4) — exercises the assembled money path offline -----------
#
# test_price_* above pin the PARSE contract; these lock the five just-applied money-path fixes
# end-to-end by driving run_price with synthetic calibration + forecast rows. The DB engine,
# settings, and queries.latest are mocked (mirroring the captured_range pattern) so the pure
# math (link.predict → blend → bucket_prob → EV → Kelly) runs for real with no Postgres.

_F_TO_K_OFFSET = 273.15


def _f_to_kelvin(temp_f: float) -> float:
    """Inverse of strata.kelvin_to_fahrenheit, so synthetic members land at a known °F."""
    return (temp_f - 32.0) * 5.0 / 9.0 + _F_TO_K_OFFSET


def _cal_row(
    model: str,
    *,
    crps_oos: float | None = 1.0,
    n_train: int | None = 500,
    pool_level: str = "own:city",
    mean_intercept: float | None = 0.0,
    mean_slope: float | None = 1.0,
    var_intercept: float | None = 1.0,
    var_slope: float | None = 0.0,
    sigma_floor: float | None = 0.5,
) -> dict:
    """An identity-link calibration row: μ = mean_f, σ = 1.0 (var_intercept=1, floor=0.5)."""
    return {
        "model": model,
        "mean_intercept": mean_intercept,
        "mean_slope": mean_slope,
        "var_intercept": var_intercept,
        "var_slope": var_slope,
        "sigma_floor": sigma_floor,
        "crps_oos": crps_oos,
        "n_train": n_train,
        "pool_level": pool_level,
    }


def _forecast_rows(model: str, temp_f: float, members: int = 2) -> list[dict]:
    """`members` identical members for `model` at `temp_f` (var_f = 0 → deterministic)."""
    return [
        {"model": model, "temp_kelvin": _f_to_kelvin(temp_f)} for _ in range(members)
    ]


def _patch_price_db(
    monkeypatch: pytest.MonkeyPatch,
    *,
    forecasts: list[dict],
    cal_rows: list[dict],
    afd_rows: list[dict] | None = None,
    cap: float = 0.025,
) -> None:
    """Patch get_engine/get_settings/queries.latest so run_price runs offline (D-16)."""
    afd = afd_rows or []

    def _fake_latest(bind, table, where=None):  # noqa: ANN001
        if table == "forecasts":
            return forecasts
        if table == "calibration_params":
            return cal_rows
        if table == "observations":
            return afd
        raise AssertionError(f"unexpected table {table!r}")

    monkeypatch.setattr(cli, "get_engine", lambda: object())
    monkeypatch.setattr(
        cli, "get_settings", lambda: type("S", (), {"max_position_fraction": cap})()
    )
    monkeypatch.setattr("weatherquant.db.queries.latest", _fake_latest)


def _price_args(ticker: str | None = "KXHIGHNY-62-63", market_mid: str = "0.1"):
    argv = ["price", "--city", "NYC", "--date", "2026-06-12", "--market-mid", market_mid]
    if ticker is not None:
        argv += ["--ticker", ticker]
    return cli.build_parser().parse_args(argv)


def test_run_price_positive_edge_ev_and_stake_agree_in_sign(
    monkeypatch: pytest.MonkeyPatch,
):
    """A positive-edge bucket prints EV > 0 AND stake > 0 — they agree in sign (locks WR-01).

    μ_blend = 62.5, σ = 1.0 puts ~68% mass in [62, 63] vs a mocked mid of 0.10, so the edge
    (and the Kelly stake sized on the SAME shrunk p_used) is unambiguously positive.
    """
    _patch_price_db(
        monkeypatch,
        forecasts=_forecast_rows("hrrr", 62.5),
        cal_rows=[_cal_row("hrrr")],
    )
    result = cli.run_price(_price_args())
    bucket = result["buckets"][0]
    assert bucket["ev"] > 0.0
    assert bucket["stake_fraction"] > 0.0  # same sign as EV (WR-01)


def test_run_price_null_n_train_raises(monkeypatch: pytest.MonkeyPatch):
    """A NULL n_train on the sizing model fails loud instead of silently zeroing (locks WR-02)."""
    _patch_price_db(
        monkeypatch,
        forecasts=_forecast_rows("hrrr", 62.5),
        cal_rows=[_cal_row("hrrr", n_train=None)],
    )
    with pytest.raises(SystemExit, match="NULL n_train"):
        cli.run_price(_price_args())


def test_run_price_drops_model_with_null_emos_param(monkeypatch: pytest.MonkeyPatch):
    """A model with ANY NULL EMOS param drops out of the blend (locks WR-04)."""
    _patch_price_db(
        monkeypatch,
        forecasts=_forecast_rows("hrrr", 62.5) + _forecast_rows("gfs", 62.5),
        cal_rows=[_cal_row("hrrr"), _cal_row("gfs", mean_intercept=None)],
    )
    result = cli.run_price(_price_args())
    assert result["models"] == ["hrrr"]  # gfs dropped on NULL mean_intercept (D-03)


def test_run_price_min_ramp_model_chosen_deterministically(
    monkeypatch: pytest.MonkeyPatch,
):
    """Under tied weights the smallest-sufficiency model sizes the blend, order-independent (WR-05).

    Two models with identical CRPS (⇒ tied weights) and identical (μ, σ) but different n_train.
    The stake must reflect the THIN model's ramp regardless of forecast-row insertion order;
    argmax(weights) used to leak iteration order here.
    """
    thin = _cal_row("aaa", n_train=15)  # ramp = 15/30 = 0.5 (the min)
    thick = _cal_row("zzz", n_train=500)  # ramp = 1.0

    def _stake(forecasts):
        _patch_price_db(
            monkeypatch, forecasts=forecasts, cal_rows=[thin, thick]
        )
        return cli.run_price(_price_args())["buckets"][0]["stake_fraction"]

    fwd = _forecast_rows("aaa", 62.5) + _forecast_rows("zzz", 62.5)
    rev = _forecast_rows("zzz", 62.5) + _forecast_rows("aaa", 62.5)
    stake_fwd = _stake(fwd)
    stake_rev = _stake(rev)
    assert stake_fwd == stake_rev  # deterministic under tied weights (WR-05)

    # And it is the MIN-ramp (thin, n_train=15) model that sets the haircut, not the thick one.
    from weatherquant import price as pricing

    prob = pricing.bucket_prob(62.5, 1.0, 61.5, 63.5)
    pu = pricing.p_used(prob, 0.1)
    expected = pricing.stake_fraction(
        pu, 0.1, pricing.exact_fee(1, 0.1), 1.0, 15, "own:city", False, cap=0.025
    )
    assert stake_fwd == pytest.approx(expected)
