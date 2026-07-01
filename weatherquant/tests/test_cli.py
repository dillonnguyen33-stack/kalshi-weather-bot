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

    monkeypatch.setattr(cli.ingest, "get_engine", _fake_get_engine)
    monkeypatch.setattr(cli.ingest.orchestrator, "ingest_range", _fake_ingest_range)
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


def test_live_subcommand_parses_with_no_args():
    """`weatherquant live` is a registered no-arg subcommand (dispatch wiring contract)."""
    args = cli.build_parser().parse_args(["live"])
    assert args.command == "live"


def test_live_serve_starts_then_shuts_down_scheduler(monkeypatch: pytest.MonkeyPatch):
    """_serve start()s the scheduler and, when its forever-wait is cancelled, shutdown()s it."""
    import asyncio

    from weatherquant.cli.live import _serve

    fake = type("S", (), {"started": False, "stopped": False, "get_jobs": lambda self: []})()
    fake.start = lambda: setattr(fake, "started", True)
    fake.shutdown = lambda wait=False: setattr(fake, "stopped", True)
    monkeypatch.setattr("weatherquant.cli.live.build_scheduler", lambda: fake)

    # _serve waits forever; wait_for cancels it after it has started, triggering the finally.
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(asyncio.wait_for(_serve(), timeout=0.05))
    assert fake.started and fake.stopped


def test_run_live_returns_zero_on_interrupt(monkeypatch: pytest.MonkeyPatch):
    """Ctrl-C (KeyboardInterrupt out of asyncio.run) is a clean stop → exit code 0."""

    def _interrupt(coro):
        coro.close()  # avoid 'coroutine was never awaited' noise
        raise KeyboardInterrupt

    monkeypatch.setattr("weatherquant.cli.live.asyncio.run", _interrupt)
    assert cli.run_live(object()) == 0


def test_build_scheduler_registers_per_model_jobs():
    """build_scheduler wires AsyncIOScheduler per model cadence WITHOUT starting (D-15)."""
    from weatherquant.scheduler import build_scheduler

    scheduler = build_scheduler()
    jobs = scheduler.get_jobs()
    # >=4 jobs registered (HRRR/NBM hourly + GFS/GEFS 00/06/12/18Z, plus obs/AFD cadence).
    assert len(jobs) >= 4
    # The scheduler is configured but NOT started (unit-testable).
    assert scheduler.running is False


def test_scheduler_grib_job_ingests_both_leads_per_city(monkeypatch: pytest.MonkeyPatch):
    """The live grib job ingests lead 0 AND lead 24 for every city (feeds the traded horizon)."""
    import asyncio

    from weatherquant import scheduler as sched
    from weatherquant.registry import CITIES

    calls: list[tuple[str, int]] = []

    async def _fake_ingest(bind, model, city, cycle, *, mode, lead):  # noqa: ANN001
        calls.append((city, lead))

    monkeypatch.setattr(sched, "get_engine", lambda: object())
    monkeypatch.setattr(sched.orchestrator, "ingest_cycle", _fake_ingest)
    asyncio.run(sched._ingest_grib_all_cities("hrrr", 1))

    assert sched.LIVE_LEADS == (0, 24)
    for city in CITIES:
        assert (city, 0) in calls, city
        assert (city, 24) in calls, city


def test_scheduler_lead24_failure_does_not_sink_lead0(monkeypatch: pytest.MonkeyPatch):
    """A missing fxx=24 (e.g. HRRR hourly) degrades+logs; the lead-0 write still happens for all cities."""
    import asyncio

    from weatherquant import scheduler as sched
    from weatherquant.registry import CITIES

    calls: list[tuple[str, int]] = []

    async def _fake_ingest(bind, model, city, cycle, *, mode, lead):  # noqa: ANN001
        calls.append((city, lead))
        if lead == 24:
            raise RuntimeError("no fxx=24 on this cycle")

    monkeypatch.setattr(sched, "get_engine", lambda: object())
    monkeypatch.setattr(sched.orchestrator, "ingest_cycle", _fake_ingest)
    # Must NOT raise despite every lead-24 call failing.
    asyncio.run(sched._ingest_grib_all_cities("hrrr", 1))
    assert all((city, 0) in calls for city in CITIES)


def test_scheduler_jobs_have_explicit_misfire_policy():
    """Each live job sets an explicit misfire_grace_time + coalesce so a late/overrunning cycle
    is handled deliberately, not silently dropped under apscheduler's 1-second default."""
    from weatherquant.scheduler import build_scheduler

    jobs = build_scheduler().get_jobs()
    assert jobs
    for job in jobs:
        assert job.misfire_grace_time is not None and job.misfire_grace_time >= 60
        assert job.coalesce is True


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
    assert args.lead == 24  # default: the calibrated/traded 24h-ahead high
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

    monkeypatch.setattr(cli.pricing, "get_engine", lambda: object())
    monkeypatch.setattr(
        cli.pricing, "get_settings", lambda: type("S", (), {"max_position_fraction": cap})()
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


# --- paper subcommand (05-04) — the REAL live-book midpoint loop closer ----------------------
#
# run_paper feeds the REAL reflection-derived live-book midpoint into the Phase-4 money path
# (closing the D-08/D-16 loop) and persists the snapshot + (possibly partial) fill via the
# audited path — paper only, no real order. The WS/REST snapshot is MOCKED with a scripted book;
# the DB + settings + signer are mocked so run_paper runs offline.

from datetime import datetime, timedelta  # noqa: E402

from weatherquant.market import client as ws_client  # noqa: E402
from weatherquant.registry import get_city  # noqa: E402
from weatherquant.time import settlement_window  # noqa: E402

_PAPER_DATE = date(2026, 6, 18)
_PAPER_TICKER = "KXHIGHNY-62-63"


def _scripted_paper_book(event_time: datetime) -> dict:
    """A scripted two-sided book: yes bid 40¢, no bid 56¢ → yes ask 44¢ → mid 42¢ (0.42)."""
    return {
        "type": "orderbook_snapshot",
        "seq": 7,
        "ticker": _PAPER_TICKER,
        "yes": [[40, 200]],
        "no": [[56, 200]],
        "event_time": event_time,
    }


def _expected_reflection_mid(book: dict) -> float:
    """Independently recompute the reflection-derived mid the loop MUST close on (T-05-19)."""
    from weatherquant.market import reflect

    best_yes_bid = max(int(p) for p, _ in book["yes"])
    best_yes_ask = reflect.yes_ask_levels(book)[0][0]  # 100 - best_no_bid
    return ((best_yes_bid + best_yes_ask) / 2.0) / 100.0


def _patch_paper(
    monkeypatch: pytest.MonkeyPatch,
    *,
    book: dict,
    forecasts: list[dict],
    cal_rows: list[dict],
    cap: float = 0.025,
) -> dict:
    """Patch the snapshot fetch, persist seams, DB + settings + signer so run_paper runs offline.

    Returns a ``captured`` dict recording the persist_snapshot / persist_fill call kwargs.
    """
    captured: dict = {"snapshots": [], "fills": []}

    async def _fake_fetch(http, signer, ticker, *, rest_host=None):  # noqa: ANN001
        return book

    def _fake_persist_snapshot(bind, **kw):  # noqa: ANN001
        captured["snapshots"].append(kw)
        return 1

    def _fake_persist_fill(bind, **kw):  # noqa: ANN001
        captured["fills"].append(kw)
        return 1

    def _fake_latest(bind, table, where=None):  # noqa: ANN001
        if table == "forecasts":
            return forecasts
        if table == "calibration_params":
            return cal_rows
        if table == "observations":
            return []
        raise AssertionError(f"unexpected table {table!r}")

    class _Signer:
        @classmethod
        def from_settings(cls, settings):  # noqa: ANN001
            return cls()

        def sign(self, method, path):  # noqa: ANN001
            return {}

    _settings = lambda: type(  # noqa: E731
        "S", (), {"max_position_fraction": cap, "execution_mode": "paper"}
    )()
    monkeypatch.setattr(cli.paper, "get_engine", lambda: object())
    monkeypatch.setattr(cli.paper, "get_settings", _settings)
    # _blend_distribution lives in cli.pricing and reads get_settings()/cap there.
    monkeypatch.setattr(cli.pricing, "get_settings", _settings)
    monkeypatch.setattr(cli.paper, "fetch_snapshot", _fake_fetch)
    monkeypatch.setattr(cli.paper, "persist_snapshot", _fake_persist_snapshot)
    monkeypatch.setattr(cli.paper, "persist_fill", _fake_persist_fill)
    monkeypatch.setattr(cli.paper, "KalshiSigner", _Signer)
    monkeypatch.setattr("weatherquant.db.queries.latest", _fake_latest)
    return captured


def _paper_args(
    ticker: str = _PAPER_TICKER,
    demo: bool = False,
    watch: bool = False,
    max_duration: int | None = None,
):
    argv = ["paper", "--city", "NYC", "--date", "2026-06-18", "--ticker", ticker]
    if demo:
        argv.append("--demo")
    if watch:
        argv.append("--watch")
    if max_duration is not None:
        argv += ["--max-duration", str(max_duration)]
    return cli.build_parser().parse_args(argv)


def test_paper_watch_flag_defaults_off_and_parses_on():
    """`--watch` is store_true: absent -> False (single-shot default), present -> True."""
    assert _paper_args().watch is False
    assert _paper_args(watch=True).watch is True


def test_paper_max_duration_parses_and_has_positive_default():
    """`--max-duration` parses to the typed seconds; the default is a positive safety cap."""
    assert _paper_args(max_duration=600).max_duration == 600
    default_cap = _paper_args().max_duration
    assert isinstance(default_cap, int)
    assert default_cap > 0


def test_paper_max_duration_rejects_non_positive_at_parse_time():
    """A non-positive `--max-duration` raises SystemExit at parse time (T-051-01)."""
    parser = cli.build_parser()
    for bad in ("0", "-5"):
        with pytest.raises(SystemExit) as excinfo:
            parser.parse_args(
                ["paper", "--city", "NYC", "--date", "2026-06-18", "--ticker", _PAPER_TICKER,
                 "--max-duration", bad]
            )
        assert excinfo.value.code != 0


def test_run_paper_produces_midpoint_fed_ev_and_paper_fill(monkeypatch: pytest.MonkeyPatch):
    """(a) run_paper produces a midpoint-fed EV/stake and a paper fill (no real order)."""
    win = settlement_window(get_city("NYC"), _PAPER_DATE)
    event_time = win.end_utc - timedelta(minutes=5)  # inside the CLV window
    book = _scripted_paper_book(event_time)
    _patch_paper(
        monkeypatch,
        book=book,
        forecasts=_forecast_rows("hrrr", 62.5),
        cal_rows=[_cal_row("hrrr")],
    )
    result = cli.run_paper(_paper_args())
    assert result["midpoint"] == pytest.approx(_expected_reflection_mid(book))
    assert result["ev"] > 0.0  # μ=62.5 σ=1 vs a 0.42 mid → positive edge
    assert result["stake"] > 0.0
    assert result["fill"] is not None  # a paper fill was simulated
    assert result["fill"]["count"] >= 1


def test_run_paper_loop_closure_value_into_p_used(monkeypatch: pytest.MonkeyPatch):
    """(b) LOOP-CLOSURE VALUE: the market_mid passed into price.p_used EQUALS the reflected mid (T-05-19)."""
    win = settlement_window(get_city("NYC"), _PAPER_DATE)
    event_time = win.end_utc - timedelta(minutes=5)
    book = _scripted_paper_book(event_time)
    _patch_paper(
        monkeypatch,
        book=book,
        forecasts=_forecast_rows("hrrr", 62.5),
        cal_rows=[_cal_row("hrrr")],
    )

    # Capture the market_mid arg actually passed into price.p_used (the D-08/D-16 loop value).
    import weatherquant.price as pricing

    captured_mids: list[float] = []
    real_p_used = pricing.p_used

    def _spy_p_used(p_model, market_mid, *args, **kwargs):  # noqa: ANN001
        captured_mids.append(market_mid)
        return real_p_used(p_model, market_mid, *args, **kwargs)

    monkeypatch.setattr(pricing, "p_used", _spy_p_used)

    result = cli.run_paper(_paper_args())
    expected_mid = _expected_reflection_mid(book)
    # Both the returned midpoint AND the value fed into p_used equal the reflection-derived mid.
    assert result["midpoint"] == pytest.approx(expected_mid)
    assert captured_mids, "price.p_used was never called — the loop did not close"
    assert captured_mids[0] == pytest.approx(expected_mid)


def test_run_paper_single_shot_delegates_to_process_book(monkeypatch: pytest.MonkeyPatch):
    """The single-shot path runs its money tail THROUGH the shared _process_book helper (PROC-01).

    Spy-wrap the REAL helper so the assertion proves the extract is the ONE shared body (the
    watch path in Plan 02 calls the SAME helper) — not that a duplicated inline body happens to
    match. The behavior assertions in test_run_paper_produces_midpoint_fed_ev_and_paper_fill stay
    the regression net; this only proves the call wiring.
    """
    win = settlement_window(get_city("NYC"), _PAPER_DATE)
    event_time = win.end_utc - timedelta(minutes=5)
    book = _scripted_paper_book(event_time)
    _patch_paper(
        monkeypatch,
        book=book,
        forecasts=_forecast_rows("hrrr", 62.5),
        cal_rows=[_cal_row("hrrr")],
    )

    real_process_book = cli.paper._process_book
    calls: list[dict] = []

    def _spy_process_book(bind, **kwargs):  # noqa: ANN001
        calls.append(kwargs)
        return real_process_book(bind, **kwargs)

    monkeypatch.setattr(cli.paper, "_process_book", _spy_process_book)

    result = cli.run_paper(_paper_args())

    assert len(calls) == 1, "single-shot run_paper must call _process_book exactly once"
    # The caller owns the event-time SOURCE and hands it to the helper as a param (D-08).
    assert calls[0]["event_time"] == event_time
    assert calls[0]["book"] is book
    # The delegated result is surfaced unchanged through run_paper's return dict.
    assert result["midpoint"] == pytest.approx(_expected_reflection_mid(book))
    assert result["fill"] is not None


def test_paper_snapshot_cadence_is_strictly_finer_than_clv_window():
    """PAP-04: the target snapshot cadence must stay strictly finer than the CLV closing window so a
    feed-driven loop honouring it never leaves the window silently sparse (moved off the import path).
    """
    from weatherquant.cli.paper import PAPER_SNAPSHOT_CADENCE_SECONDS
    from weatherquant.market.clv import CLV_WINDOW_MINUTES

    assert PAPER_SNAPSHOT_CADENCE_SECONDS < CLV_WINDOW_MINUTES * 60


def test_run_paper_cadence_sufficiency_persists_snapshot_in_closing_window(
    monkeypatch: pytest.MonkeyPatch,
):
    """(c) PAP-04 CADENCE SUFFICIENCY: a book change inside the CLV window persists a snapshot there."""
    from weatherquant.market import clv

    win = settlement_window(get_city("NYC"), _PAPER_DATE)
    window_start = win.end_utc - timedelta(minutes=clv.CLV_WINDOW_MINUTES)
    event_time = win.end_utc - timedelta(minutes=clv.CLV_WINDOW_MINUTES // 2)  # inside window
    book = _scripted_paper_book(event_time)
    captured = _patch_paper(
        monkeypatch,
        book=book,
        forecasts=_forecast_rows("hrrr", 62.5),
        cal_rows=[_cal_row("hrrr")],
    )

    cli.run_paper(_paper_args())

    # The cadence is strictly finer than the window, so the in-window book change lands >= 1
    # persisted snapshot whose available_at is inside [end - CLV_WINDOW, end).
    in_window = [
        kw
        for kw in captured["snapshots"]
        if window_start <= kw["available_at"] < win.end_utc
    ]
    assert len(in_window) >= 1
    assert cli.PAPER_SNAPSHOT_CADENCE_SECONDS < clv.CLV_WINDOW_MINUTES * 60


def test_run_paper_persists_supporting_top_of_book_volume(monkeypatch: pytest.MonkeyPatch):
    """(e) CORR-MED-3: persisted volume = the top-of-book size SUPPORTING the yes mid.

    The persisted ``volume`` must be ``min(best_yes_bid_size, best_no_bid_size)`` — the
    top-of-book two-sided size behind the persisted yes mid (the yes-ask supporting size IS the
    best-no-bid size, per the reflection seam) — NOT the summed two-sided union depth. With a
    THIN yes touch (size 10) and a DEEP no touch (size 1000) the weight is ``min(10, 1000) == 10``,
    never ``10 + 1000 == 1010``.
    """
    win = settlement_window(get_city("NYC"), _PAPER_DATE)
    event_time = win.end_utc - timedelta(minutes=5)
    book = {
        "type": "orderbook_snapshot",
        "seq": 7,
        "ticker": _PAPER_TICKER,
        "yes": [[40, 10]],  # THIN at touch
        "no": [[56, 1000]],  # DEEP at touch (opposite side)
        "event_time": event_time,
    }
    captured = _patch_paper(
        monkeypatch,
        book=book,
        forecasts=_forecast_rows("hrrr", 62.5),
        cal_rows=[_cal_row("hrrr")],
    )

    cli.run_paper(_paper_args())

    assert captured["snapshots"], "no snapshot was persisted"
    persisted_volume = captured["snapshots"][0]["volume"]
    assert persisted_volume == 10  # min(10, 1000) — the supporting size, NOT 1010 (the old union)


def test_run_paper_volume_invariant_to_opposite_side_depth_growth(
    monkeypatch: pytest.MonkeyPatch,
):
    """(f) CORR-MED-3 INVARIANCE: growing ONLY the opposite (no) side's depth leaves volume fixed.

    The weight tracks the priced (supporting) side. A no-heavy snapshot must NOT get a larger
    weight on its thinly-supported yes-mid: with the no touch grown from 1000 to 10000 the
    persisted volume stays at the supporting ``min(10, ...) == 10``.
    """
    win = settlement_window(get_city("NYC"), _PAPER_DATE)
    event_time = win.end_utc - timedelta(minutes=5)
    book = {
        "type": "orderbook_snapshot",
        "seq": 7,
        "ticker": _PAPER_TICKER,
        "yes": [[40, 10]],  # supporting size unchanged
        "no": [[56, 10000]],  # opposite-side depth grown 10x
        "event_time": event_time,
    }
    captured = _patch_paper(
        monkeypatch,
        book=book,
        forecasts=_forecast_rows("hrrr", 62.5),
        cal_rows=[_cal_row("hrrr")],
    )

    cli.run_paper(_paper_args())

    assert captured["snapshots"], "no snapshot was persisted"
    # Invariant: only the opposite side grew; the supporting min is still the yes-bid size 10.
    assert captured["snapshots"][0]["volume"] == 10


def test_run_paper_unknown_city_rejected_before_any_io(capsys: pytest.CaptureFixture):
    """(d) An unknown city is rejected by _city_type BEFORE any I/O (ASVS V5)."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["paper", "--city", "ZZZ", "--date", "2026-06-18", "--ticker", _PAPER_TICKER])
    assert excinfo.value.code != 0
    err = capsys.readouterr().err
    assert "ZZZ" in err


# --- CRIT-1 regression guard: run_paper end-to-end on the REAL fetch_snapshot shape ----------
#
# This is the standing guard for AUDIT-CRIT-1: it drives the REAL market.client.fetch_snapshot
# (mocking ONLY the httpx transport, NOT cli.fetch_snapshot) so run_paper exercises the actual
# producer's output shape — the snapshot self-stamps event_time at the fetch site, so
# _snapshot_event_time resolves and run_paper no longer aborts. The fixture must NEVER again
# hand-inject event_time (the divergence that masked the gap). With Task 1 reverted (no stamp)
# this test FAILS with SystemExit "no usable event time".


class _RealShapeResponse:
    """An httpx-response-shaped object: headers (Date) + raise_for_status + json (orderbook_fp)."""

    def __init__(self, payload, headers):
        self._payload = payload
        self.headers = dict(headers)

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _RealShapeAsyncClient:
    """An httpx.AsyncClient-shaped async context manager returning a fixed REST orderbook GET.

    Drives the REAL fetch_snapshot through its actual transport seam (await http.get(...)) so
    the test exercises the production parse + observed-instant stamp, not a stubbed fetch.
    """

    def __init__(self, payload, headers):
        self._payload = payload
        self._headers = headers

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, *, params=None, headers=None):
        return _RealShapeResponse(self._payload, self._headers)


def test_run_paper_end_to_end_real_fetch_snapshot(monkeypatch: pytest.MonkeyPatch):
    """CRIT-1 GUARD: run_paper runs end-to-end on the REAL fetch_snapshot output (no injected event_time).

    The HTTP transport is mocked; cli.fetch_snapshot stays REAL. The snapshot carries a Date
    header inside the CLV closing window for the NYC settlement date, so fetch_snapshot stamps
    event_time at the fetch site and the money path (pricing -> fill -> persist) completes.
    """
    import httpx

    win = settlement_window(get_city("NYC"), _PAPER_DATE)
    # An observed instant inside the half-open CLV closing window, formatted as an RFC-1123
    # Date header (the producer's observed-instant source under D-08).
    observed = (win.end_utc - timedelta(minutes=5)).replace(microsecond=0)
    date_header = observed.strftime("%a, %d %b %Y %H:%M:%S GMT")

    # The REAL orderbook_fp payload shape (dollar-string pairs), NOT a pre-stamped book: yes bid
    # 40c, no bid 56c -> reflected yes ask 44c -> mid 42c (0.42), a positive edge vs mu=62.5.
    real_payload = {
        "orderbook_fp": {
            "seq": 7,
            "yes_dollars": [["0.40", "200"]],
            "no_dollars": [["0.56", "200"]],
        }
    }

    captured = _patch_paper(
        monkeypatch,
        book={},  # unused: the fetch patch below is OVERRIDDEN to the real client
        forecasts=_forecast_rows("hrrr", 62.5),
        cal_rows=[_cal_row("hrrr")],
    )
    # OVERRIDE the _patch_paper fetch stub: restore the REAL fetch_snapshot and mock ONLY the
    # httpx transport so run_paper exercises the production producer shape (no injected event_time).
    monkeypatch.setattr(cli.paper, "fetch_snapshot", ws_client.fetch_snapshot)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda *a, **k: _RealShapeAsyncClient(real_payload, {"Date": date_header})
    )

    result = cli.run_paper(_paper_args())

    # run_paper ran end-to-end on the REAL shape (it did NOT SystemExit): a midpoint-fed EV and
    # a persisted snapshot whose available_at is the fetch-stamped observed instant.
    assert result["midpoint"] == pytest.approx(0.42)
    assert result["ev"] > 0.0
    assert captured["snapshots"], "no snapshot persisted — the real-shape money path did not run"
    persisted_at = captured["snapshots"][0]["available_at"]
    assert persisted_at == observed  # the fetch-site stamp, not an injected/back-dated time


def test_run_paper_refuses_live_mode(monkeypatch: pytest.MonkeyPatch):
    """run_paper does NOT run in validated live mode (no order-submission path, D-15/T-05-14)."""
    win = settlement_window(get_city("NYC"), _PAPER_DATE)
    book = _scripted_paper_book(win.end_utc - timedelta(minutes=5))
    _patch_paper(
        monkeypatch,
        book=book,
        forecasts=_forecast_rows("hrrr", 62.5),
        cal_rows=[_cal_row("hrrr")],
    )
    # Flip settings to live — the paper simulator must refuse before any fill.
    monkeypatch.setattr(
        cli.paper,
        "get_settings",
        lambda: type("S", (), {"max_position_fraction": 0.025, "execution_mode": "live"})(),
    )
    with pytest.raises(SystemExit, match="live"):
        cli.run_paper(_paper_args())


# --- verify subcommand (06-05 Task 3) — parse parity + non-zero-exit propagation -------------
#
# These pin the verify subcommand's CONTRACT (the heavy proof/drift orchestration is exercised in
# the verify unit/integration suites): the subparser mirrors the calibrate/paper selectors, the
# validators reject a bad city BEFORE the body (ASVS V5), and cli.main propagates run_verify's int
# exit code (a drift breach therefore yields a non-zero process exit — SYS-02).


def test_verify_subcommand_parses_selectors_and_flags():
    """A valid `verify` invocation parses the window/lead/monitor/window-days/out-dir + selectors."""
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "verify", "--city", "NYC", "--model", "hrrr",
            "--start", "2026-06-01", "--end", "2026-06-15",
            "--lead", "0", "--monitor", "--window-days", "14", "--out-dir", "out",
        ]
    )
    assert args.command == "verify"
    assert args.city == "NYC"
    assert args.model == "hrrr"
    assert args.start == date(2026, 6, 1)
    assert args.end == date(2026, 6, 15)
    assert args.monitor is True
    assert args.window_days == 14
    assert args.out_dir == "out"


def test_verify_unknown_city_rejected_before_body(capsys: pytest.CaptureFixture):
    """An unknown `verify --city` is rejected by argparse via _city_type — ASVS V5 / T-06-19."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["verify", "--city", "ZZZ", "--model", "hrrr"])
    assert excinfo.value.code != 0
    assert "ZZZ" in capsys.readouterr().err


def test_main_verify_propagates_nonzero_exit(monkeypatch: pytest.MonkeyPatch):
    """cli.main returns run_verify's int — a non-zero (drift breach) propagates (SYS-02)."""
    import sys

    cli_main = sys.modules["weatherquant.cli.main"]  # the MODULE (cli.main is the re-exported fn)
    monkeypatch.setattr(cli_main, "run_verify", lambda args: 3)
    # --start/--end are now MANDATORY on the verify subparser (CR-04) even for the monitor flag.
    rc = cli.main(
        ["verify", "--city", "NYC", "--model", "hrrr",
         "--start", "2026-06-01", "--end", "2026-06-15", "--monitor"]
    )
    assert rc == 3  # propagated unchanged (NOT collapsed to 0 like the count-dict branches)


def test_main_verify_returns_zero_on_clean_run(monkeypatch: pytest.MonkeyPatch):
    """A clean verify run returns 0 through cli.main (the verdict PASS/FAIL lives in the artifact)."""
    import sys

    cli_main = sys.modules["weatherquant.cli.main"]  # the MODULE (cli.main is the re-exported fn)
    monkeypatch.setattr(cli_main, "run_verify", lambda args: 0)
    # --start/--end are now MANDATORY on the verify subparser (CR-04).
    rc = cli.main(
        ["verify", "--city", "NYC", "--model", "hrrr", "--start", "2026-06-01", "--end", "2026-06-15"]
    )
    assert rc == 0


# --- verify verdict-path window + fail-closed OOS-slice guards (06-06 Task 3, CR-04) ----------


def _verify_args(**overrides):
    """A minimal argparse.Namespace for run_verify's verdict path (monitor off)."""
    import argparse

    base = dict(
        city="NYC", model="hrrr", all_cities=False, all_models=False,
        start=date(2026, 6, 1), end=date(2026, 6, 15), lead=0, monitor=False,
        window_days=30, out_dir="reports",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _patch_verify_settings(monkeypatch, *, oos_start, oos_end):
    """Patch cli.verify's get_engine/get_settings seams with a fake Settings carrying the OOS knob."""
    from weatherquant import cli as _cli

    settings = type(
        "S", (), {"verify_phase3_oos_start": oos_start, "verify_phase3_oos_end": oos_end}
    )()
    monkeypatch.setattr(_cli.verify, "get_engine", lambda: object())
    monkeypatch.setattr(_cli.verify, "get_settings", lambda: settings)
    return settings


def test_verify_verdict_rejects_inverted_window(monkeypatch: pytest.MonkeyPatch):
    """The verdict path raises SystemExit when end <= start (CR-04)."""
    from weatherquant.cli.verify import run_verify

    _patch_verify_settings(monkeypatch, oos_start=date(2025, 1, 1), oos_end=date(2025, 6, 1))
    args = _verify_args(start=date(2026, 6, 15), end=date(2026, 6, 1))  # inverted
    with pytest.raises(SystemExit):
        run_verify(args)


def test_verify_verdict_rejects_unset_oos_slice(monkeypatch: pytest.MonkeyPatch):
    """The verdict path raises SystemExit when the Phase-3 OOS knob is UNSET (fail-closed, CR-04)."""
    from weatherquant.cli.verify import run_verify

    _patch_verify_settings(monkeypatch, oos_start=None, oos_end=None)  # knob unset
    with pytest.raises(SystemExit):
        run_verify(_verify_args())


def test_verify_verdict_rejects_empty_oos_slice(monkeypatch: pytest.MonkeyPatch):
    """The verdict path raises SystemExit when the OOS slice is empty/inverted (CR-04)."""
    from weatherquant.cli.verify import run_verify

    _patch_verify_settings(monkeypatch, oos_start=date(2025, 6, 1), oos_end=date(2025, 1, 1))
    with pytest.raises(SystemExit):
        run_verify(_verify_args())


def test_verify_verdict_overlapping_oos_slice_fails_loud(monkeypatch: pytest.MonkeyPatch):
    """A NON-EMPTY OOS slice overlapping the Gate-1 window raises ValueError on the real path (CR-04).

    The OOS slice is fed into walk_forward, which runs assert_window_disjoint FIRST (before any
    ledger access, so bind=object() never executes a query) — proving the guard is no longer a no-op.
    """
    from weatherquant.cli.verify import run_verify

    # OOS slice [2026-06-10, 2026-06-20) overlaps the Gate-1 window [2026-06-01, 2026-06-15).
    _patch_verify_settings(monkeypatch, oos_start=date(2026, 6, 10), oos_end=date(2026, 6, 20))
    with pytest.raises(ValueError):
        run_verify(_verify_args())


def test_verify_verdict_records_oos_slice_in_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """The resolved non-empty OOS slice is recorded into the GATE1-VERDICT artifact (CR-04)."""
    import json

    from weatherquant.cli.verify import run_verify

    _patch_verify_settings(monkeypatch, oos_start=date(2025, 1, 1), oos_end=date(2025, 6, 1))
    # bind=None so walk_forward returns no records (disjoint slice → guard passes, empty verdict).
    from weatherquant import cli as _cli

    monkeypatch.setattr(_cli.verify, "get_engine", lambda: None)
    rc = run_verify(_verify_args(out_dir=str(tmp_path)))
    assert rc == 0
    payload = json.loads((tmp_path / "GATE1-VERDICT.json").read_text())
    assert payload["oos_slice"] == ["2025-01-01", "2025-06-01"]


# --- 06-07 Task 2/3: CR-01 real CRPS + ledger ROI/CLV + the pinned not_scored sentinel ---------
#
# These prove the verdict scores the PRE-REGISTERED metrics, not the deleted (wq-v3)*(2o-1) proxy:
# a fast unit proof of the no-fills not_scored sentinel + the crps_blend wiring, then a seeded
# NON-EMPTY-ledger integration proof (fills + closing snapshots + a tail-high day).


def test_roi_clv_not_scored_sentinel_fails_loud_when_no_fills():
    """CR-01/T-06-20: no fills → roi/clv map to the pinned (0.0,0.0) sentinel, gate1 FAILs cleanly."""
    from weatherquant.cli import verify as verify_mod
    from weatherquant.verify import gate1, metrics

    # bind=None → the fills read returns nothing → not_scored. No exception, the pinned sentinel.
    (roi_ci, clv_ci), not_scored = verify_mod._roi_clv_cis(
        None, "NYC", "hrrr", date(2026, 6, 1), date(2026, 6, 15), metrics
    )
    assert roi_ci == (0.0, 0.0) == clv_ci == verify_mod._NOT_SCORED_CI
    assert not_scored is True
    # The full five-key dict keeps the exact set so gate1_passes does NOT raise — it returns False.
    cis = {
        "brier": (-0.1, -0.05), "crps": (-0.1, -0.05), "ece": (-0.1, -0.05),
        "roi": roi_ci, "clv": clv_ci,
    }
    assert set(cis) == gate1.GATE1_METRICS  # the key-set assertion will not raise
    assert gate1.gate1_passes(cis) is False  # a FAIL VERDICT, not an exception (roi/clv 0.0 > 0 fails)
    # And the not_scored sentinel is NOT an inf/NaN that would break metric_passes' comparisons.
    assert gate1.metric_passes("roi", *roi_ci) is False
    assert gate1.metric_passes("clv", *clv_ci) is False


def test_not_scored_renders_not_scored_fail_in_verdict_artifact(tmp_path):
    """CR-01: a not_scored roi/clv renders 'not scored / FAIL' verbatim in GATE1-VERDICT.md."""
    from weatherquant.verify import report

    cis = {
        "brier": (-0.1, -0.05), "crps": (-0.1, -0.05), "ece": (-0.1, -0.05),
        "roi": (0.0, 0.0), "clv": (0.0, 0.0),
    }
    verdict = {"passed": False, "seed": 0, "not_scored": {"roi": True, "clv": True}}
    report.render_reports([], cis, verdict, out_dir=str(tmp_path), seed=0, coverage=[])
    md = (tmp_path / "GATE1-VERDICT.md").read_text()
    assert "not scored / FAIL" in md
    # The numeric CIs for the not_scored metrics must NOT appear as a real computation.
    assert "| roi | 0.0 | 0.0 |" not in md
    assert "| clv | 0.0 | 0.0 |" not in md


def test_crps_score_fn_is_real_crps_blend_delta_not_proxy():
    """CR-01: _crps_score_fn returns crps_blend(wq) - crps_blend(v3), NOT the deleted proxy."""
    import numpy as np

    from weatherquant.cli import verify as verify_mod
    from weatherquant.verify import metrics
    from weatherquant.verify.backtest import PairedRecord

    day = date(2026, 7, 10)
    # Two records on ONE day, each carrying the per-record predictive Gaussians 06-06 populates.
    recs = [
        PairedRecord(
            day=day, city="KXHIGHNY", bucket=(85, 85), wq_prob=0.6, v3_prob=0.55, o_i=1,
            wq_mu=85.0, wq_sigma=2.0, v3_mu=84.0, v3_sigma=3.0, y=85.5,
        ),
        PairedRecord(
            day=day, city="KXHIGHNY", bucket=(86, 86), wq_prob=0.3, v3_prob=0.35, o_i=0,
            wq_mu=85.0, wq_sigma=2.0, v3_mu=84.0, v3_sigma=3.0, y=85.5,
        ),
    ]
    score_fn = verify_mod._crps_score_fn(recs, metrics)
    got = score_fn([day])
    # Hand-computed: crps_blend pools BOTH records' (mu, sigma, y) for each arm.
    wq_mu = np.array([85.0, 85.0])
    wq_sigma = np.array([2.0, 2.0])
    y = np.array([85.5, 85.5])
    v3_mu = np.array([84.0, 84.0])
    v3_sigma = np.array([3.0, 3.0])
    expected = metrics.crps_blend(wq_mu, wq_sigma, y) - metrics.crps_blend(v3_mu, v3_sigma, y)
    assert got == pytest.approx(expected)
    # The deleted proxy mean((wq-v3)*(2o-1)) would be a DIFFERENT number on these records.
    proxy = float(np.mean((np.array([0.6, 0.3]) - np.array([0.55, 0.35])) * (2.0 * np.array([1.0, 0.0]) - 1.0)))
    assert got != pytest.approx(proxy)


def test_crps_score_fn_fails_loud_without_predictive_params():
    """CR-01: a scored record missing the predictive params raises — never silently a proxy."""
    from weatherquant.cli import verify as verify_mod
    from weatherquant.verify import metrics
    from weatherquant.verify.backtest import PairedRecord

    day = date(2026, 7, 10)
    rec = PairedRecord(day=day, city="KXHIGHNY", bucket=(85, 85), wq_prob=0.6, v3_prob=0.55, o_i=1)
    score_fn = verify_mod._crps_score_fn([rec], metrics)
    with pytest.raises(ValueError):
        score_fn([day])


# --- 06-07 Task 3: seeded NON-EMPTY-ledger end-to-end proof CR-01/CR-03 are closed -------------
#
# The verifier flagged that the prior integration tests ran against an EMPTY DB, so the real
# scoring path (fills/closing-snapshots ROI/CLV + the tail-day coverage) was never exercised
# end-to-end. THESE seeded tests are the standing proof — a green unit suite alone is NOT evidence.

from datetime import datetime as _dt, timedelta as _td, timezone as _tz  # noqa: E402

_VER_F_TO_K = lambda f: (f - 32.0) * 5.0 / 9.0 + 273.15  # noqa: E731 - test-only inverse K→°F seam


def _seed_jja_forecasts_observations(conn, *, tail_high: bool = False, tail_day=None):
    """Seed June+July NYC/hrrr forecasts+observations on the PRODUCTION no-look-ahead path (06-10).

    Decision-day settled obs are stamped at ``settlement_window(get_city("NYC"), d).end_utc`` (the
    realistic LST settlement instant on/after day d), NOT back-dated to 2026-01-01; forecasts are
    stamped at ``d-2d`` 00:00 UTC (strictly before the ``d-1d`` 00:00 UTC cutoff). Seeds JUNE 01..30
    + JULY 01..31 (the JJA season) so >= N_MIN JJA pairs survive the cutoff for the scored July
    window and the July month-fit is retained (the same feasibility arithmetic as the backtest
    seeder — June pairs all settle before a mid-July cutoff). When ``tail_high``, ``tail_day``'s obs
    is a TAIL high (far above the interior ±4σ range) so CR-03's open-upper bucket + tail_settlement
    coverage-log is exercised on the production path.
    """
    from weatherquant.ingest.writer import insert_forecast, insert_observation

    base_f = 85.0

    def _seed_month(month: int, n_days: int):
        for i in range(n_days):
            d = date(2026, month, 1 + i)
            members_f = [base_f - 2.0, base_f, base_f + 2.0]  # 3 members → s2 = 8/3 > 0
            cycle = _dt(d.year, d.month, d.day, 0, tzinfo=_tz.utc)
            # Forecasts at d-2d 00:00 UTC — strictly before the d-1d cutoff (own forecast survives).
            fc_avail = _dt(d.year, d.month, d.day, tzinfo=_tz.utc) - _td(days=2)
            for member, mf in enumerate(members_f):
                insert_forecast(
                    conn, city="NYC", target_date=d, model="hrrr", lead=0, member=member,
                    temp_kelvin=_VER_F_TO_K(mf), cycle=cycle,
                    station_lat=40.779, station_lon=-73.969, grid_distance_m=1000.0,
                    available_at=fc_avail,
                )
            high = (base_f + 100.0) if (tail_high and d == tail_day) else (base_f + 0.5)
            # Decision-day obs at the realistic LST SETTLEMENT instant (excluded from the < d-1d
            # training read, present in the < d+2d outcome read).
            obs_avail = settlement_window(get_city("NYC"), d).end_utc
            insert_observation(
                conn, city="NYC", target_date=d, source="asos", daily_high_f=high,
                available_at=obs_avail,
            )

    _seed_month(6, 30)  # June, JJA — season-retention fill so >= N_MIN JJA pairs survive the cutoff
    _seed_month(7, 31)  # July, JJA — the scored month


def _seed_verdict_ledger(conn, *, tail_high: bool):
    """Seed a NON-EMPTY July ledger: forecasts+observations, fills, closing snapshots, a tail day.

    06-10: re-stamped to the PRODUCTION no-look-ahead path — decision-day obs at settlement, forecasts
    at d-2d, with June+July (JJA) seeded so >= N_MIN JJA pairs survive the cutoff and the July
    month-fit is retained for the scored window (the verdict path scores rather than no_month_fit).
    The last replayed day's observation is a TAIL high when ``tail_high`` so CR-03's open-upper bucket
    + tail_settlement coverage-log is exercised. Returns the (ticker, day, avg_price_cents,
    closing_mid_cents) for the seeded fill so the caller can hand-compute ROI/CLV.
    """
    from weatherquant.ingest.writer import insert_fill, insert_market_snapshot

    _seed_jja_forecasts_observations(conn, tail_high=tail_high, tail_day=date(2026, 7, 12))

    # Seed ONE fill on 2026-07-10 (in the Gate-1 window) on the 85°F bucket + its closing snapshots.
    avail = _dt(2026, 1, 1, tzinfo=_tz.utc)  # fills/snapshots back-stamp is fine (not a settled obs)
    fill_day = date(2026, 7, 10)
    ticker = "KXHIGHNY-85-86"  # the [85,86] bucket → span [84.5, 86.5) (settled high 85.5 → YES)
    fill_event = _dt(2026, 7, 10, 12, tzinfo=_tz.utc)
    insert_fill(
        conn, ticker=ticker, trade_id="t1", side="yes", price=40, count=2, fee=1,
        is_maker=False, event_time=fill_event,
        detail={"avg_price_cents": 40.0}, available_at=avail,
    )
    # Closing-window snapshots inside [end - CLV_WINDOW_MINUTES, end) for the fill's settlement day.
    from weatherquant.market.clv import CLV_WINDOW_MINUTES

    win = settlement_window(get_city("NYC"), fill_day)
    closing_mid_cents = 50.0  # a single in-window snapshot → vol-weighted mid = 50.0c
    t_in = win.end_utc - _td(minutes=CLV_WINDOW_MINUTES - 5)
    insert_market_snapshot(
        conn, ticker=ticker, snapshot_for=t_in.isoformat(),
        best_yes_bid=49, best_no_bid=49, mid=closing_mid_cents, volume=100, seq=None,
        detail={"raw": "book"}, available_at=t_in,
    )
    return ticker, fill_day, 40.0, closing_mid_cents


def _seed_multiday_fills(conn, *, fills_spec, ticker="KXHIGHNY-85-86", snapless_lst_days=frozenset()):
    """Seed forecasts/observations for July + one fill per spec entry on its own LST settlement day.

    ``fills_spec`` is a list of ``(event_time_utc, side, avg_price_cents)`` tuples. The default
    ``ticker`` is the ``KXHIGHNY-85-86`` bucket; the interior observation high (85.5°F) settles YES,
    so a YES buy is a win and a NO buy is a loss — letting the side="no" e2e tests assert the NO
    orientation. Pass a NON-containing bucket (e.g. ``KXHIGHNY-50-51``, span [49.5, 51.5)) so the
    settled high 85.5 is OUTSIDE it → the bucket settles NO → a NO buy WINS. Closing snapshots are
    seeded per distinct LST settlement day (a single 50c in-window snap → mid 50c). Returns the list
    of resolved LST settlement days (one per fill).

    06-10: forecasts/observations are seeded via the shared PRODUCTION-path seeder (decision-day obs
    at settlement, forecasts at d-2d) — the fills are settled by ``_settle_window_fills`` via a plain
    ``queries.latest`` obs read (no as-of cutoff), so realistic obs stamping is correct here.
    """
    from weatherquant.ingest.writer import insert_fill, insert_market_snapshot
    from weatherquant.market.clv import CLV_WINDOW_MINUTES

    _seed_jja_forecasts_observations(conn)

    avail = _dt(2026, 1, 1, tzinfo=_tz.utc)  # fills/snapshots back-stamp (not a settled obs)
    city = get_city("NYC")
    seeded_snap_days: set = set()
    lst_days: list = []
    for n, (event_time, side, avg_price) in enumerate(fills_spec):
        insert_fill(
            conn, ticker=ticker, trade_id=f"m{n}", side=side, price=int(round(avg_price)),
            count=1, fee=0, is_maker=False, event_time=event_time,
            detail={"avg_price_cents": float(avg_price)}, available_at=avail,
        )
        # LST settlement day = (event_time + offset).date() (offset = -5h for NYC) — the WR-04 key.
        lst_day = (event_time + _td(hours=city.std_offset_hours)).date()
        lst_days.append(lst_day)
        # ``snapless_lst_days`` leaves a day's closing window EMPTY (no snapshot) — the WR-1 case.
        if lst_day not in seeded_snap_days and lst_day not in snapless_lst_days:
            win = settlement_window(city, lst_day)
            t_in = win.end_utc - _td(minutes=CLV_WINDOW_MINUTES - 5)
            insert_market_snapshot(
                conn, ticker=ticker, snapshot_for=t_in.isoformat(),
                best_yes_bid=49, best_no_bid=49, mid=50.0, volume=100, seq=None,
                detail={"raw": "book"}, available_at=t_in,
            )
            seeded_snap_days.add(lst_day)
    return lst_days


@pytest.mark.integration
def test_verdict_scores_roi_clv_off_real_fills_real_ci_width(pg_conn):
    """CR-01 / GAP-2 (seeded e2e): multi-LST-day window → a REAL bootstrap CI with positive width.

    With >= _MIN_FILL_DAYS_FOR_CI distinct LST fill-days of VARYING profitability, ROI/CLV are real
    day-block bootstrap intervals — never a zero-width point. A degenerate point CI is the GAP-2
    defect this regression locks shut.
    """
    from weatherquant.cli import verify as verify_mod
    from weatherquant.verify import metrics

    # Three distinct interior July LST days, YES buys at varying prices → varying ROI per block.
    spec = [
        (_dt(2026, 7, 5, 12, tzinfo=_tz.utc), "yes", 30.0),
        (_dt(2026, 7, 7, 12, tzinfo=_tz.utc), "yes", 50.0),
        (_dt(2026, 7, 9, 12, tzinfo=_tz.utc), "yes", 70.0),
    ]
    _seed_multiday_fills(pg_conn, fills_spec=spec)
    (roi_ci, clv_ci), not_scored = verify_mod._roi_clv_cis(
        pg_conn, "KXHIGHNY", "hrrr", date(2026, 7, 1), date(2026, 7, 13), metrics
    )
    assert not_scored is False, "multi-LST-day ledger must be scored, not the sentinel"
    assert roi_ci[1] > roi_ci[0], "ROI CI must have strictly positive width (never a zero-width point)"
    assert clv_ci[1] > clv_ci[0], "CLV CI must have strictly positive width (never a zero-width point)"


@pytest.mark.integration
def test_verdict_two_distinct_days_identical_roi_maps_to_not_scored_sentinel(pg_conn):
    """CR-01 (BLOCKER regression): 2 distinct LST fill-days with IDENTICAL per-day ROI → not_scored.

    The ``_MIN_FILL_DAYS_FOR_CI = 2`` floor is necessary but NOT sufficient. With exactly two
    distinct days A and B whose per-day ROI is identical (two SAME-price YES buys on the same bucket
    that both settle YES), every paired day-block resample ({A,A}, {A,B}, {B,B}) yields the SAME
    pooled ROI — so ``np.percentile(deltas, [2.5, 97.5])`` returns ``roi_lo == roi_hi``: a zero-width
    "interval" at a profitable point that formerly read as a Gate-1 PASS (``ci_lo > 0``). A
    degenerate CI cannot honestly exclude zero, so it must map to the PINNED not_scored sentinel and
    FAIL, never manufacture a money-go PASS on n=2-with-identical-ROI.
    """
    from weatherquant.cli import verify as verify_mod
    from weatherquant.verify import gate1, metrics

    # Two distinct interior July LST days, SAME-price YES buys on the SAME YES-settling bucket →
    # identical per-day ROI on every block → a zero-width bootstrap CI.
    spec = [
        (_dt(2026, 7, 5, 12, tzinfo=_tz.utc), "yes", 40.0),
        (_dt(2026, 7, 9, 12, tzinfo=_tz.utc), "yes", 40.0),
    ]
    _seed_multiday_fills(pg_conn, fills_spec=spec)
    (roi_ci, clv_ci), not_scored = verify_mod._roi_clv_cis(
        pg_conn, "KXHIGHNY", "hrrr", date(2026, 7, 1), date(2026, 7, 13), metrics
    )
    assert not_scored is True, (
        "two distinct fill-days with IDENTICAL per-day ROI yield a zero-width CI — it must map to "
        "the not_scored sentinel, NEVER a degenerate point PASS (CR-01 BLOCKER)"
    )
    assert roi_ci == (0.0, 0.0) == clv_ci
    # And the full conjunctive gate FAILs cleanly on the pinned sentinel (not an exception).
    cis = {
        "brier": (-0.1, -0.05), "crps": (-0.1, -0.05), "ece": (-0.1, -0.05),
        "roi": roi_ci, "clv": clv_ci,
    }
    assert gate1.gate1_passes(cis) is False


@pytest.mark.integration
def test_verdict_zero_capital_fill_fails_closed_not_mid_bootstrap_crash(pg_conn):
    """WR-03: a window with a zero-cost (avg_price_cents == 0) fill fails CLOSED, never crashes.

    ``roi_from_fills`` raises ValueError when the pooled capital deployed is non-positive
    (``total_entry <= 0``). Inside the day-block bootstrap, a resample drawing only zero-cost fills
    would make that raise PROPAGATE out of ``paired_day_block_ci``, aborting the entire Gate-1 run
    mid-bootstrap rather than declining to score. A money gate must fail closed (decline), not
    crash. With a 0c fill present, ``_roi_clv_cis`` must map ROI/CLV to the pinned not_scored
    sentinel and return cleanly — NEVER raise.
    """
    from weatherquant.cli import verify as verify_mod
    from weatherquant.verify import metrics

    # Two distinct LST days (so the distinct-day floor is cleared) but one fill deploys ZERO capital
    # (avg_price_cents == 0). The bootstrap must not be allowed to raise on a zero-capital resample.
    spec = [
        (_dt(2026, 7, 5, 12, tzinfo=_tz.utc), "yes", 0.0),   # zero-cost fill (no capital deployed)
        (_dt(2026, 7, 9, 12, tzinfo=_tz.utc), "yes", 40.0),
    ]
    _seed_multiday_fills(pg_conn, fills_spec=spec)
    # Must return cleanly (fail closed), never raise out of the bootstrap.
    (roi_ci, clv_ci), not_scored = verify_mod._roi_clv_cis(
        pg_conn, "KXHIGHNY", "hrrr", date(2026, 7, 1), date(2026, 7, 13), metrics
    )
    assert not_scored is True, "a zero-capital fill window must fail closed → not_scored (WR-03)"
    assert roi_ci == (0.0, 0.0) == clv_ci


@pytest.mark.integration
def test_verdict_empty_closing_window_fails_closed_not_mid_bootstrap_crash(pg_conn):
    """WR-1 (the WR-03 twin): a positive-capital fill whose closing window has NO snapshot fails
    CLOSED, never crashes. ``mean_clv`` → ``clv_cents`` → ``vol_weighted_mid([])`` raises on an empty
    closing window; inside the day-block bootstrap that raise would PROPAGATE out of
    ``paired_day_block_ci`` and abort the entire Gate-1 verdict. The WR-03 capital guard does not
    cover this (capital is positive), so ``_roi_clv_cis`` must map ROI/CLV to the pinned not_scored
    sentinel and return cleanly — NEVER raise.
    """
    from weatherquant.cli import verify as verify_mod
    from weatherquant.verify import metrics

    # Two distinct LST days, both positive capital (clears WR-03 + the distinct-day floor), but the
    # 2026-07-09 day's closing window is left EMPTY (no snapshot persisted in the final CLV minutes).
    snapless_lst_day = (
        _dt(2026, 7, 9, 12, tzinfo=_tz.utc) + _td(hours=get_city("NYC").std_offset_hours)
    ).date()
    spec = [
        (_dt(2026, 7, 5, 12, tzinfo=_tz.utc), "yes", 40.0),
        (_dt(2026, 7, 9, 12, tzinfo=_tz.utc), "yes", 40.0),  # closing window left empty
    ]
    _seed_multiday_fills(pg_conn, fills_spec=spec, snapless_lst_days={snapless_lst_day})
    # Must return cleanly (fail closed), never raise out of the bootstrap.
    (roi_ci, clv_ci), not_scored = verify_mod._roi_clv_cis(
        pg_conn, "KXHIGHNY", "hrrr", date(2026, 7, 1), date(2026, 7, 13), metrics
    )
    assert not_scored is True, "an empty closing window must fail closed → not_scored (WR-1)"
    assert roi_ci == (0.0, 0.0) == clv_ci


@pytest.mark.integration
def test_verdict_single_fill_day_maps_to_not_scored_sentinel(pg_conn):
    """CR-01 / GAP-2 core regression: a SINGLE profitable fill-day can NEVER flip Gate-1 to PASS.

    Below _MIN_FILL_DAYS_FOR_CI distinct LST fill-days, ROI/CLV map to the pinned (0.0, 0.0)
    not_scored sentinel — a lone profitable fill is not a passing CI.
    """
    from weatherquant.cli import verify as verify_mod
    from weatherquant.verify import metrics

    spec = [(_dt(2026, 7, 5, 12, tzinfo=_tz.utc), "yes", 40.0)]  # exactly ONE LST fill-day
    _seed_multiday_fills(pg_conn, fills_spec=spec)
    (roi_ci, clv_ci), not_scored = verify_mod._roi_clv_cis(
        pg_conn, "KXHIGHNY", "hrrr", date(2026, 7, 1), date(2026, 7, 13), metrics
    )
    assert not_scored is True, "a single fill-day must map to the not_scored sentinel, never a pass"
    assert roi_ci == (0.0, 0.0) == clv_ci


@pytest.mark.integration
def test_verdict_no_fill_side_settles_yes_is_scored_as_a_loss_end_to_end(pg_conn):
    """T-06-09-T2 (seeded e2e): side='no' fills that settle YES are scored as LOSSES via threaded sides.

    The interior high settles YES, so a NO buy LOSES. A side-BLIND ROI would have scored these NO
    buys as YES WINS (positive); the threaded side must instead score them as losses.

    CR-01 interaction (why this is two assertions, not a "negative CI"): a pure all-NO-loss ledger
    is DEGENERATE — every losing block has ROI exactly ``-1.0`` (payoff 0 → (0 - entry)/entry = -1
    for ANY price), so the day-block bootstrap CI is zero-width and CORRECTLY maps to the not_scored
    sentinel (a zero-width "interval" cannot honestly exclude zero, even on the favorable side). We
    therefore prove the orientation in two parts:

    1. the verdict path declines to score the degenerate ledger (not_scored is True); and
    2. at the metric layer, the SETTLED inputs produced by ``_settle_window_fills`` feed
       ``roi_from_fills`` to a per-block ROI of exactly ``-1.0`` (a LOSS) — never the ``+1.0`` a
       side-blind reading (NO scored as a YES win on this YES-settling bucket) would yield.

    Part 2 is the real orientation proof (not_scored alone is sign-agnostic: a uniform all-WIN
    ledger is also degenerate). The NON-degenerate positive complement is covered by
    ``test_verdict_roi_side_no_fill_settles_no_is_a_win`` (NO wins at varying prices → a real CI).
    """
    from weatherquant.cli import verify as verify_mod
    from weatherquant.verify import metrics

    spec = [
        (_dt(2026, 7, 5, 12, tzinfo=_tz.utc), "no", 30.0),
        (_dt(2026, 7, 7, 12, tzinfo=_tz.utc), "no", 50.0),
        (_dt(2026, 7, 9, 12, tzinfo=_tz.utc), "no", 70.0),
    ]
    _seed_multiday_fills(pg_conn, fills_spec=spec)
    (roi_ci, clv_ci), not_scored = verify_mod._roi_clv_cis(
        pg_conn, "KXHIGHNY", "hrrr", date(2026, 7, 1), date(2026, 7, 13), metrics
    )
    # 1. The all-NO-loss ledger is degenerate (uniform -1.0 per block) → not_scored (CR-01).
    assert not_scored is True, "a uniform all-loss NO ledger is a zero-width CI → not_scored (CR-01)"
    assert roi_ci == (0.0, 0.0) == clv_ci

    # 2. Orientation proof at the metric layer: the SETTLED NO fills score to a LOSS (-1.0), the
    # exact opposite of the +1.0 a side-blind YES-win reading would produce. This is what reaches
    # roi_from_fills through the verdict path (same seams _roi_clv_cis uses).
    from weatherquant.db import queries

    all_fills = list(queries.latest(pg_conn, "fills"))
    window_fills = [
        f for f in all_fills
        if verify_mod._fill_in_window(f, "NYC", date(2026, 7, 1), date(2026, 7, 13))
    ]
    assert len(window_fills) == 3, "all three seeded NO fills are in the window"
    settled_yes, _clv_fills, _snaps, sides = verify_mod._settle_window_fills(
        pg_conn, window_fills, "NYC"
    )
    assert all(s == "sell" for s in sides), "side='no' normalizes to 'sell' (the NO mirror)"
    assert all(sy is True for sy in settled_yes), "the interior bucket settles YES on every day"
    roi = metrics.roi_from_fills(window_fills, settled_yes, sides)
    assert roi == pytest.approx(-1.0), (
        "side='no' buys on a YES-settling bucket must score a LOSS (-1.0), never the +1.0 a "
        "side-blind YES-win reading would produce (sides threaded into roi_from_fills)"
    )


@pytest.mark.integration
def test_verdict_roi_side_no_fill_settles_no_is_a_win(pg_conn):
    """T-06-09-T2 (seeded e2e, 06-10): a side='no' fill on a NO-settling bucket is CREDITED as a win.

    The complement of the loss test: here the fills sit on a NON-containing bucket (KXHIGHNY-50-51,
    span [49.5, 51.5)) — the settled interior high (85.5°F) is OUTSIDE it, so the bucket settles NO
    and a NO buy WINS. The positive ROI CI proves the NO win is credited end-to-end through
    ``_settle_window_fills`` → ``roi_from_fills(sides=...)`` (a side-blind ROI would have scored these
    NO buys on a non-containing bucket as YES losses → a negative CI).
    """
    from weatherquant.cli import verify as verify_mod
    from weatherquant.verify import metrics

    # NO buys (cheap) on a bucket the settled high does NOT land in → each NO position wins.
    spec = [
        (_dt(2026, 7, 5, 12, tzinfo=_tz.utc), "no", 30.0),
        (_dt(2026, 7, 7, 12, tzinfo=_tz.utc), "no", 35.0),
        (_dt(2026, 7, 9, 12, tzinfo=_tz.utc), "no", 40.0),
    ]
    _seed_multiday_fills(pg_conn, fills_spec=spec, ticker="KXHIGHNY-50-51")
    (roi_ci, clv_ci), not_scored = verify_mod._roi_clv_cis(
        pg_conn, "KXHIGHNY", "hrrr", date(2026, 7, 1), date(2026, 7, 13), metrics
    )
    assert not_scored is False
    # NO-settling bucket + NO buys → every block wins → the whole ROI CI is above zero.
    assert roi_ci[0] > 0, "a side='no' buy that settles NO must be credited as a win (sides threaded)"


@pytest.mark.integration
def test_verdict_no_fill_clv_magnitude_is_no_denominated(pg_conn):
    """WR-02 (seeded e2e): a NO fill's CLV is scored NO-denominated (100 - yes_mid) - price.

    The existing NO e2e tests only assert the ROI CI SIGN — never the CLV MAGNITUDE. A NO
    ('sell'-normalized) fill records its NO-contract price; its closing value is the NO mid
    ``100 - yes_mid``. The seeded closing snapshot has a YES mid of 50.0c, so the NO mid is 50.0c
    and a NO buy at 30/35/40c has a per-fill CLV of ``50 - price`` = 20 / 15 / 10c (all POSITIVE —
    bought the NO cheap relative to its close).

    The OLD (buggy) orientation differenced the NO price against the YES mid and flipped the sign:
    ``-(50 - 30) = -20`` — a units mismatch that inverted the sign and mis-scaled the magnitude.
    Asserting the EXACT magnitude pins the NO-denominated CLV so the orientation can't regress.
    """
    from weatherquant.cli import verify as verify_mod
    from weatherquant.db import queries
    from weatherquant.verify import metrics

    spec = [
        (_dt(2026, 7, 5, 12, tzinfo=_tz.utc), "no", 30.0),
        (_dt(2026, 7, 7, 12, tzinfo=_tz.utc), "no", 35.0),
        (_dt(2026, 7, 9, 12, tzinfo=_tz.utc), "no", 40.0),
    ]
    _seed_multiday_fills(pg_conn, fills_spec=spec, ticker="KXHIGHNY-50-51")

    # Exact per-fill / mean magnitude via the same settle seams _roi_clv_cis uses.
    all_fills = list(queries.latest(pg_conn, "fills"))
    window_fills = [
        f for f in all_fills
        if verify_mod._fill_in_window(f, "NYC", date(2026, 7, 1), date(2026, 7, 13))
    ]
    assert len(window_fills) == 3
    _settled, clv_fills, snaps, sides = verify_mod._settle_window_fills(
        pg_conn, window_fills, "NYC"
    )
    assert all(s == "sell" for s in sides), "side='no' normalizes to 'sell'"
    # YES mid 50.0c → NO mid 50.0c → CLV = 50 - price = {20, 15, 10}; mean = 15.0c (NO-denominated).
    mean_clv = metrics.mean_clv(clv_fills, snaps, sides)
    assert mean_clv == pytest.approx(15.0), (
        "a NO fill's CLV must be NO-denominated (100 - yes_mid) - price = 50 - price; the OLD "
        "YES-denominated orientation would give -(yes_mid - price), a sign-inverted units bug (WR-02)"
    )

    # And the bootstrap CLV CI is strictly POSITIVE (bought the NO cheap), never the negative the
    # old orientation produced.
    (roi_ci, clv_ci), not_scored = verify_mod._roi_clv_cis(
        pg_conn, "KXHIGHNY", "hrrr", date(2026, 7, 1), date(2026, 7, 13), metrics
    )
    assert not_scored is False
    assert clv_ci[0] > 0, "the NO-denominated CLV CI is strictly positive (NO bought cheap)"


@pytest.mark.integration
def test_verdict_roi_clv_block_key_is_lst_settlement_day_not_utc_date(pg_conn):
    """T-06-09-T3 / WR-04 (seeded e2e): the bootstrap block key is the LST settlement day, not UTC date.

    A near-boundary fill at 2026-07-11 02:00 UTC belongs to LST day 2026-07-10 (NYC offset -5h:
    02:00 - 5h = 21:00 on 07-10). Paired with TWO other fills clearly inside 07-10 and 07-08, the
    distinct LST-day count is 2 (07-08, 07-10) → SCORED. If the block key were the raw UTC date, the
    boundary fill would key on 07-11, giving 3 distinct UTC dates — a DIFFERENT grouping. We assert
    the near-boundary fill JOINS the 07-10 block (same as the clearly-in-07-10 fill), proven by the
    scored verdict landing on 2 LST blocks (not 3 UTC dates).
    """
    from weatherquant.cli import verify as verify_mod
    from weatherquant.verify import metrics

    boundary = _dt(2026, 7, 11, 2, 0, tzinfo=_tz.utc)  # UTC date 07-11, LST settlement day 07-10
    in_0710 = _dt(2026, 7, 10, 12, tzinfo=_tz.utc)      # clearly inside LST day 07-10
    in_0708 = _dt(2026, 7, 8, 12, tzinfo=_tz.utc)       # a second distinct LST day
    spec = [
        (in_0708, "yes", 30.0),
        (in_0710, "yes", 50.0),
        (boundary, "yes", 70.0),
    ]
    lst_days = _seed_multiday_fills(pg_conn, fills_spec=spec)
    # The boundary fill resolves to the SAME LST day as in_0710 (07-10), NOT its UTC date (07-11).
    assert lst_days == [date(2026, 7, 8), date(2026, 7, 10), date(2026, 7, 10)]
    (roi_ci, clv_ci), not_scored = verify_mod._roi_clv_cis(
        pg_conn, "KXHIGHNY", "hrrr", date(2026, 7, 1), date(2026, 7, 13), metrics
    )
    # Two distinct LST blocks (07-08, 07-10) >= _MIN_FILL_DAYS_FOR_CI → scored with a real CI.
    assert not_scored is False, "two distinct LST fill-days must score (boundary fill joins 07-10)"
    assert roi_ci[1] >= roi_ci[0]


@pytest.mark.integration
def test_verdict_roi_clv_not_scored_sentinel_fails_loud_when_no_fills(pg_conn):
    """CR-01 (seeded e2e): with NO fills for the window, roi/clv map to (0.0,0.0) → gate1 FAIL."""
    from weatherquant.cli import verify as verify_mod
    from weatherquant.verify import gate1, metrics

    _seed_verdict_ledger(pg_conn, tail_high=False)  # seeds a fill on 2026-07-10 ...
    # ... but score a DIFFERENT (later, fill-free) window so the fills ledger is empty for it.
    (roi_ci, clv_ci), not_scored = verify_mod._roi_clv_cis(
        pg_conn, "KXHIGHNY", "hrrr", date(2026, 7, 20), date(2026, 7, 25), metrics
    )
    assert not_scored is True
    assert roi_ci == (0.0, 0.0) == clv_ci
    cis = {
        "brier": (-0.1, -0.05), "crps": (-0.1, -0.05), "ece": (-0.1, -0.05),
        "roi": roi_ci, "clv": clv_ci,
    }
    assert gate1.gate1_passes(cis) is False  # FAIL verdict, not an exception


@pytest.mark.integration
def test_tail_high_day_is_scored_and_coverage_logged(pg_conn):
    """CR-03 (seeded e2e): a tail-high day is o_i=1 in the open-upper bucket + coverage-logged."""
    from weatherquant.verify import backtest

    _seed_verdict_ledger(pg_conn, tail_high=True)  # 2026-07-12 settles at a tail high (185°F)
    records, coverage = backtest.walk_forward(
        pg_conn, "KXHIGHNY", "hrrr", lead=0,
        start=date(2026, 7, 10), end=date(2026, 7, 13),
        oos_slice=(date(2025, 1, 1), date(2025, 6, 1)),
    )
    tail_day = date(2026, 7, 12)
    # The tail day is coverage-logged tail_settlement (auditable, never a silent o_i=0 drop).
    assert any(
        e.get("day") == tail_day and e.get("reason") == "tail_settlement" for e in coverage
    ), "the tail-high day must be coverage-logged tail_settlement (CR-03/D-09)"
    # The tail day's records are STILL scored, and exactly ONE bucket (the open-upper) is its YES.
    tail_records = [r for r in records if r.day == tail_day and r.excluded_reason is None]
    assert tail_records, "the tail day must still be scored (not dropped)"
    assert sum(r.o_i for r in tail_records) == 1, "exactly one YES bucket for the tail high"
