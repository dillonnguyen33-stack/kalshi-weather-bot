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


def test_build_scheduler_registers_per_model_jobs():
    """build_scheduler wires AsyncIOScheduler per model cadence WITHOUT starting (D-15)."""
    from weatherquant.scheduler import build_scheduler

    scheduler = build_scheduler()
    jobs = scheduler.get_jobs()
    # >=4 jobs registered (HRRR/NBM hourly + GFS/GEFS 00/06/12/18Z, plus obs/AFD cadence).
    assert len(jobs) >= 4
    # The scheduler is configured but NOT started (unit-testable).
    assert scheduler.running is False


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
    rc = cli.main(
        ["verify", "--city", "NYC", "--model", "hrrr", "--monitor"]
    )
    assert rc == 3  # propagated unchanged (NOT collapsed to 0 like the count-dict branches)


def test_main_verify_returns_zero_on_clean_run(monkeypatch: pytest.MonkeyPatch):
    """A clean verify run returns 0 through cli.main (the verdict PASS/FAIL lives in the artifact)."""
    import sys

    cli_main = sys.modules["weatherquant.cli.main"]  # the MODULE (cli.main is the re-exported fn)
    monkeypatch.setattr(cli_main, "run_verify", lambda args: 0)
    rc = cli.main(["verify", "--city", "NYC", "--model", "hrrr"])
    assert rc == 0
