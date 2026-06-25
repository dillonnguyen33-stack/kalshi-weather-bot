"""``weatherquant verify`` — run the Gate-1 paired proof + render the verdict (D-12 / VER-* / SYS-02).

D-12 (verify subtree-local; CLI subcommand): ``run_verify`` is the terminal Gate-1 entry point. The
DEFAULT path orchestrates the walk-forward paired backtest → paired day-block bootstrap CIs → the
pre-registered conjunctive gate verdict → the report render, freezing/loading the pre-registration
BEFORE the verdict is computed (D-08/D-13 anti-p-hacking — the freeze must precede the result). The
``--monitor`` path runs the rolling reliability-error drift monitor and PROPAGATES its (possibly
non-zero) exit code so cron/systemd surfaces a calibration breach (SYS-02) — unlike the count-dict
subcommands, ``main.py`` returns ``run_verify``'s int directly.

``get_engine`` / ``get_settings`` are imported INTO this module's namespace (the test seam — the run
body resolves ``cli.verify.get_engine`` / ``cli.verify.get_settings``); the heavy
``weatherquant.verify`` imports stay LAZY in the body so ``--help`` and the ingest path never pay for
matplotlib or the metric core. ``run_verify`` returns an ``int`` exit code so a drift breach (or a
Gate-1 FAIL) can propagate non-zero (SYS-02).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

# Imported INTO the module namespace as the test seam (the run body resolves
# ``cli.verify.get_engine`` / ``cli.verify.get_settings``).
from weatherquant.db.engine import get_engine, get_settings

from ._args import _resolve_cities, _resolve_models

logger = logging.getLogger(__name__)

# The five pre-registered Gate-1 metrics + their direction sign-map (mirrors gate1; frozen in the
# pre-registration so the bar can't be moved after the results are seen, D-07/D-08).
_GATE1_METRICS = ("brier", "crps", "ece", "roi", "clv")
_PREREG_FILENAME = "gate1_preregistration.json"
_CI_LEVEL = 0.95
_N_RESAMPLES = 10000
_SEED = 0

# CR-01 / T-06-20 — the single PINNED not_scored sentinel CI a roi/clv metric maps to when the
# ledger has no fills for the Gate-1 window. roi/clv are HIGHER_IS_BETTER, so ``gate1.metric_passes``
# passes iff ``ci_lo > 0``; ``(0.0, 0.0)`` deterministically FAILs (``0.0 > 0`` is False) while
# keeping ``set(cis) == GATE1_METRICS`` intact so ``gate1.gate1_passes``' exact-five-key assertion
# never raises. Infinite-bound or NaN sentinels are FORBIDDEN — they would break ``metric_passes``'
# comparisons or risk the key-set/ordering assumptions. The not_scored STATUS is surfaced separately
# on the verdict so the artifact renders "not scored / FAIL" verbatim, never a numeric CI that
# implies a real computation. This is the integrity-of-verdict mitigation: a money-go PASS on
# ROI/CLV must NEVER be declarable without pricing a real fill against a real closing line.
_NOT_SCORED_CI = (0.0, 0.0)


def run_verify(args: argparse.Namespace) -> int:
    """Run the Gate-1 paired proof / drift monitor and return a process exit code (D-12 / SYS-02).

    Resolves the engine/settings via the namespace seams, then drives (lazily) either the Gate-1
    verdict path (the default) or the drift monitor (``--monitor``). The verdict path freezes/loads
    the pre-registration FIRST (D-08/D-13), runs ``verify.backtest.walk_forward``, builds the five
    pooled CIs via ``verify.bootstrap.paired_day_block_ci``, applies ``verify.gate1.gate1_passes``,
    and renders the PNGs + GATE1-VERDICT artifacts via ``verify.report.render_reports``; it returns
    ``0`` on a clean run (a Gate-1 FAIL still renders the verdict and returns ``0`` — the FAIL lives
    IN the artifact, the operator reads it). The ``--monitor`` path returns ``monitor_drift``'s int
    exit code, so a Settings-threshold breach yields a NON-ZERO process exit (SYS-02).
    """
    # Lazy so `--help` and the non-verify paths never pay the metric/report/matplotlib imports.
    from weatherquant.verify import backtest, bootstrap, drift, gate1, metrics, report

    settings = get_settings()
    bind = get_engine()
    cities = _resolve_cities(args)
    models = _resolve_models(args)

    # --- drift monitor path: propagate the non-zero exit on a breach (SYS-02) -----------------
    if getattr(args, "monitor", False):
        return drift.monitor_drift(
            bind,
            settings,
            window_days=args.window_days,
            cities=cities,
            models=models,
        )

    # --- Gate-1 verdict path -------------------------------------------------------------------
    lead = args.lead
    start = args.start
    end = args.end
    out_dir = Path(args.out_dir)

    # CR-04: fail loud on a missing/empty verdict window BEFORE any pre-registration freeze or
    # ledger access. --start/--end are argparse-required, but mirror _resolve_range's end<=start /
    # empty-window rejection so a degenerate window can never be scored.
    if start is None or end is None:
        raise SystemExit("verify: --start and --end are required for the Gate-1 verdict")
    if end <= start:
        raise SystemExit(
            f"verify: --end {end} must be strictly after --start {start} "
            "(an empty/inverted Gate-1 verdict window is rejected — CR-04)"
        )

    # CR-04: resolve the REAL Phase-3 OOS (tuning) slice FAIL-CLOSED. If either bound is unset, or
    # the slice is empty/inverted, REFUSE to run — never degrade to oos_slice=None (which would make
    # assert_window_disjoint a no-op and silently score the whole ledger incl. the tuning slice,
    # invalidating the pre-registration — D-10/D-12). A non-empty slice is threaded into
    # walk_forward so the disjointness guard fires AND recorded into the verdict for auditability.
    oos_start = settings.verify_phase3_oos_start
    oos_end = settings.verify_phase3_oos_end
    if oos_start is None or oos_end is None or oos_end <= oos_start:
        raise SystemExit(
            "verify: the Phase-3 OOS slice (verify_phase3_oos_start/end) is unset or empty — "
            "refusing to run a Gate-1 verdict without a non-empty disjoint tuning window "
            "(CR-04, D-10/D-12)"
        )
    oos_slice = (oos_start, oos_end)

    # The model that defines the comparison arm (a single model selector — the proof scores one
    # blend/model against v3). With --all-models we take the first as the pooled label.
    model = models[0]
    # One representative city for the proof window; the report still strata-splits per city below.
    city = cities[0]

    # 1. Freeze/load the pre-registration BEFORE computing the verdict (D-08/D-13 anti-p-hacking).
    out_dir.mkdir(parents=True, exist_ok=True)
    prereg_path = out_dir / _PREREG_FILENAME
    sign_map = {m: "lower" for m in ("brier", "crps", "ece")}
    sign_map.update({m: "higher" for m in ("roi", "clv")})
    spec = {
        "metrics": list(_GATE1_METRICS),
        "seed": _SEED,
        "primary_lead": lead,
        "test_window": [str(start), str(end)],
        "sign_map": sign_map,
        "ci_level": _CI_LEVEL,
        "n_resamples": _N_RESAMPLES,
    }
    if not prereg_path.exists():
        gate1.write_preregistration(prereg_path, spec)
        logger.info("verify: froze the Gate-1 pre-registration at %s (D-08)", prereg_path)
    prereg = gate1.load_preregistration(prereg_path)
    # Fail loud if this live run's params drifted from the frozen endpoint (seed/lead/window/metrics).
    gate1.assert_matches_preregistration(prereg, spec)

    # 2. Walk-forward as-of-correct paired backtest (no look-ahead, D-08). The real non-empty
    #    Phase-3 OOS slice is threaded in so assert_window_disjoint FIRES (CR-04) — an overlapping
    #    Gate-1 window fails loud rather than scoring on tuning data.
    records, coverage = backtest.walk_forward(
        bind, city, model, lead, start, end, oos_slice=oos_slice
    )
    scored = [r for r in records if r.excluded_reason is None]

    # 3. Five pooled CIs via the paired day-block bootstrap (VER-05). brier/ece keep the REAL
    #    reliability score_fn (metrics.brier / metrics.ece_equal_count on the YES probabilities vs
    #    o_i — already real, NOT a proxy); crps is the REAL paired metrics.crps_blend delta over the
    #    per-record predictive Gaussians (CR-01 — the deleted proxy never priced a real metric).
    #    Pairing is preserved by the score_fn (the same records feed both arms — RESEARCH §Pitfall 2).
    #    With no scored records the reliability/crps CIs collapse to (0, 0) (a non-passing, honest
    #    empty verdict rather than a fabricated edge).
    not_scored: dict[str, bool] = {}
    cis: dict[str, tuple[float, float]] = {}
    for name in ("brier", "ece"):
        if not scored:
            cis[name] = (0.0, 0.0)
            continue
        score_fn = _reliability_score_fn(name, scored, metrics)
        day_keys = [r.day for r in scored]
        lo, hi, _deltas = bootstrap.paired_day_block_ci(
            day_keys, score_fn, n_resamples=_N_RESAMPLES, seed=_SEED, alpha=1.0 - _CI_LEVEL
        )
        cis[name] = (lo, hi)

    # CRPS (real): metrics.crps_blend on the per-record (wq_mu, wq_sigma, y) minus
    # (v3_mu, v3_sigma, y) — the predictive Gaussians 06-06 carries on each PairedRecord. Lower is
    # better; pairing preserved. Fails loud if a scored record lacks the params (never a proxy).
    if not scored:
        cis["crps"] = (0.0, 0.0)
    else:
        crps_fn = _crps_score_fn(scored, metrics)
        day_keys = [r.day for r in scored]
        lo, hi, _deltas = bootstrap.paired_day_block_ci(
            day_keys, crps_fn, n_resamples=_N_RESAMPLES, seed=_SEED, alpha=1.0 - _CI_LEVEL
        )
        cis["crps"] = (lo, hi)

    # ROI/CLV (real off the ledger): metrics.roi_from_fills / metrics.mean_clv read the real fills +
    # market_snapshots rows for the test-window ticker(s) and settle each fill against
    # observations.daily_high_f. When the window's fills ledger is EMPTY (the common Phase-6 case —
    # the window predates live paper trading), each maps to the PINNED _NOT_SCORED_CI = (0.0, 0.0)
    # with a not_scored status — the gate FAILs explicitly ("not scored / FAIL"), never a proxy CI
    # that could read as PASS (CR-01 / T-06-19/T-06-20). Caveat: ROI/CLV are scored taker-only at
    # 1-contract sizing (RESEARCH Open Question 3 / A5) — recorded in the verdict text.
    (roi_ci, clv_ci), roi_clv_not_scored = _roi_clv_cis(
        bind, city, model, start, end, metrics
    )
    cis["roi"] = roi_ci
    cis["clv"] = clv_ci
    not_scored["roi"] = roi_clv_not_scored
    not_scored["clv"] = roi_clv_not_scored

    # 4. The pre-registered conjunctive verdict (VER-07, D-06). The full five-key CI dict (roi/clv =
    #    _NOT_SCORED_CI when unscored) is passed — gate1_passes returns False CLEANLY (a FAIL
    #    verdict), NEVER an exception, because the exact-five-key set is intact (T-06-20).
    passed = gate1.gate1_passes(cis) if scored else False

    # 5. Render the PNGs + GATE1-VERDICT.{json,md} (fragility visible — VER-02 / D-11).
    report_records = _records_for_report(scored)
    verdict = {
        "passed": passed,
        "seed": _SEED,
        "excluded_days": coverage,
        "test_window": [str(start), str(end)],
        # CR-04: record the resolved Phase-3 OOS slice into the verdict so a mis-set or surprising
        # tuning window is auditable in GATE1-VERDICT.{json,md} — never a silent no-op.
        "oos_slice": [str(oos_start), str(oos_end)],
        "primary_lead": lead,
        # CR-01 / T-06-20: the per-metric not_scored status so report.render_reports prints
        # "not scored / FAIL" verbatim for a roi/clv that mapped to the pinned (0.0, 0.0) sentinel —
        # never implying a real numeric CI was computed off an empty ledger.
        "not_scored": {name: bool(flag) for name, flag in not_scored.items() if flag},
        # ROI/CLV scoring caveat (RESEARCH Open Question 3 / A5): scored taker-only at 1-contract
        # sizing — maker rebates / size-dependent fees are out of this gate's scope.
        "roi_clv_caveat": "ROI/CLV scored taker-only at 1-contract sizing (RESEARCH OQ3/A5).",
        "preregistration": prereg,
    }
    written = report.render_reports(
        report_records, cis, verdict, out_dir=out_dir, seed=_SEED, coverage=coverage
    )
    logger.info(
        "verify: Gate-1 verdict=%s (%d scored record(s), %d excluded) — artifacts at %s",
        "PASS" if passed else "FAIL",
        len(scored),
        len(coverage),
        out_dir,
    )
    print(
        f"verify complete: Gate-1 {'PASS' if passed else 'FAIL'} "
        f"({len(written)} artifact(s) written to {out_dir})"
    )
    # The verdict (PASS or FAIL) lives in the artifact; a clean RUN returns 0. Only an operational
    # failure (or a drift breach above) is a non-zero process exit.
    return 0


def _reliability_score_fn(name, scored, metrics):
    """Build a ``score_fn(sampled_days) -> (wq - v3)`` reliability delta for ``name`` (REAL, D-06).

    Pools the paired records for the resampled day-keys and returns the weatherquant-minus-v3 delta
    on the REAL reliability metric (``metrics.brier`` / ``metrics.ece_equal_count`` on the YES
    probabilities vs the realized ``o_i``) — pairing preserved (the same buckets feed both arms).
    These were already real (NOT a proxy) before CR-01; they stay verbatim. Lower-is-better
    (``ci_hi < 0`` is WQ's edge).
    """
    import numpy as np

    by_day: dict = {}
    for r in scored:
        by_day.setdefault(r.day, []).append(r)
    scorer = metrics.brier if name == "brier" else metrics.ece_equal_count

    def score(sampled_days):
        recs = [r for d in sampled_days for r in by_day.get(d, [])]
        if not recs:
            return 0.0
        wq = np.array([r.wq_prob for r in recs], dtype=float)
        v3 = np.array([r.v3_prob for r in recs], dtype=float)
        o = np.array([float(r.o_i) for r in recs], dtype=float)
        return float(scorer(wq, o) - scorer(v3, o))

    return score


def _crps_score_fn(scored, metrics):
    """Build the REAL paired CRPS ``score_fn(sampled_days) -> crps_blend(wq) - crps_blend(v3)``.

    CR-01 / D-06 (the dead-metric fix): the verdict's CRPS metric is the REAL closed-form Gaussian
    CRPS via ``metrics.crps_blend`` on the per-record predictive Gaussians — the WQ arm's
    ``(wq_mu, wq_sigma, y)`` minus the v3 arm's ``(v3_mu, v3_sigma, y)`` — NOT the deleted
    ``(wq-v3)*(2o-1)`` probability proxy. 06-06 carries those per-record params on every scored
    ``PairedRecord``; this FAILS LOUD (raises) if any scored record lacks them rather than
    substituting a proxy. Lower-is-better (``ci_hi < 0`` is WQ's edge); pairing preserved (the same
    records feed both arms).
    """
    import numpy as np

    by_day: dict = {}
    for r in scored:
        by_day.setdefault(r.day, []).append(r)

    def score(sampled_days):
        recs = [r for d in sampled_days for r in by_day.get(d, [])]
        if not recs:
            return 0.0
        for r in recs:
            if (
                r.wq_mu is None or r.wq_sigma is None
                or r.v3_mu is None or r.v3_sigma is None or r.y is None
            ):
                raise ValueError(
                    "crps score_fn: a scored PairedRecord lacks the per-record predictive params "
                    "(wq_mu/wq_sigma/v3_mu/v3_sigma/y); CRPS must be the real crps_blend delta, "
                    "never a proxy (CR-01). 06-06 must populate them on the scored path."
                )
        y = np.array([float(r.y) for r in recs], dtype=float)
        wq_mu = np.array([float(r.wq_mu) for r in recs], dtype=float)
        wq_sigma = np.array([float(r.wq_sigma) for r in recs], dtype=float)
        v3_mu = np.array([float(r.v3_mu) for r in recs], dtype=float)
        v3_sigma = np.array([float(r.v3_sigma) for r in recs], dtype=float)
        return float(metrics.crps_blend(wq_mu, wq_sigma, y) - metrics.crps_blend(v3_mu, v3_sigma, y))

    return score


def _roi_clv_cis(bind, city, model, start, end, metrics):
    """Real ROI/CLV CIs off the ledger, or the PINNED not_scored sentinel (CR-01 / T-06-19/20).

    Reads the test-window ``fills`` + ``market_snapshots`` rows for the city's KXHIGH ticker(s) via
    ``db.queries.latest``, settles each fill against ``observations.daily_high_f`` (the bucket the
    ticker resolves to vs the settled high — RESEARCH Code Example 2), and scores
    ``metrics.roi_from_fills`` / ``metrics.mean_clv`` (CENTS, taker-only 1-contract sizing —
    RESEARCH OQ3/A5). When NO fills exist for the window (the common Phase-6 case — the window
    predates live paper trading), returns ``((_NOT_SCORED_CI, _NOT_SCORED_CI), True)``: roi/clv map
    to the PINNED ``(0.0, 0.0)`` sentinel and the caller records the ``not_scored`` status so the
    gate FAILs explicitly ("not scored / FAIL"), NEVER a proxy CI that can read as a PASS.

    Returns ``((roi_ci, clv_ci), not_scored)``: ``not_scored`` is ``True`` when the ledger had no
    fills (both CIs are ``_NOT_SCORED_CI``), ``False`` when real CIs were scored.
    """
    from weatherquant.db import queries
    from weatherquant.verify.backtest import _resolve_city_key

    city_key = _resolve_city_key(city)
    # Read the real fills ledger for the window's ticker(s). With no fills (the common Phase-6 case)
    # the metric is NOT scorable — map to the pinned sentinel and fail loud, never fabricate a CI.
    try:
        fills = list(queries.latest(bind, "fills"))
    except Exception:  # noqa: BLE001 — a ledger read failure is a not_scored, not a crash
        fills = []
    window_fills = [
        f for f in fills if _fill_in_window(f, city_key, start, end)
    ]
    if not window_fills:
        logger.info(
            "verify: no fills in the Gate-1 window for %s — ROI/CLV NOT SCORED (pinned %s, FAIL); "
            "the window predates live paper trading (CR-01).",
            city,
            _NOT_SCORED_CI,
        )
        return (_NOT_SCORED_CI, _NOT_SCORED_CI), True

    # A real non-empty fills ledger: settle each fill and score the real ROI/CLV (CENTS). The day is
    # the bootstrap block, but the fills ledger is sparse here — score the pooled point estimate as a
    # degenerate CI so the real metric is exercised end-to-end (the integration test seeds this path).
    settled_yes, clv_fills, snaps_per_fill, sides = _settle_window_fills(
        bind, window_fills, city_key
    )
    roi = metrics.roi_from_fills(window_fills, settled_yes)
    clv = metrics.mean_clv(clv_fills, snaps_per_fill, sides)
    return ((roi, roi), (clv, clv)), False


class _AvgPriceFill:
    """Minimal CLV fill adapter exposing the float ``avg_price_cents`` (D-01, never the rounded price).

    ``market.clv.clv_cents`` reads the fill's ``avg_price_cents`` ATTRIBUTE; a ledger ``fills`` row
    (a ``RowMapping``) carries it under ``detail['avg_price_cents']``. This thin adapter surfaces the
    float so the real CLV is scored off the un-rounded price (never the ±0.5c-rounded ``price`` column).
    """

    __slots__ = ("avg_price_cents",)

    def __init__(self, avg_price_cents: float) -> None:
        self.avg_price_cents = float(avg_price_cents)


def _fill_in_window(fill, city_key, start, end) -> bool:
    """True iff ``fill``'s ticker resolves to ``city_key`` and its event date is in ``[start, end)``."""
    from weatherquant.verify.backtest import _resolve_city_key

    ticker = fill.get("ticker") if hasattr(fill, "get") else getattr(fill, "ticker", None)
    if not isinstance(ticker, str):
        return False
    # The KXHIGH<SUFFIX>... ticker carries the city suffix; resolve and match the requested city.
    head = ticker.split("-", 1)[0]  # e.g. KXHIGHNY
    if _resolve_city_key(head) != city_key:
        return False
    event_time = fill.get("event_time") if hasattr(fill, "get") else getattr(fill, "event_time", None)
    if event_time is None:
        return False
    d = event_time.date()
    return start <= d < end


def _settle_window_fills(bind, window_fills, city_key):
    """Settle each window fill against the observed daily high; build the ROI/CLV scoring inputs.

    For each fill: parse its ticker to the resolved bucket, read the settled ``daily_high_f`` for
    the fill's target day via ``db.queries.latest`` (read ONLY to settle the YES/NO outcome, never a
    feature), and select the closing-window snapshots for the CLV. Returns
    ``(settled_yes, clv_fills, snaps_per_fill, sides)``: ``settled_yes[i]`` is whether the settled
    high falls in the fill's bucket (for ``roi_from_fills``); ``clv_fills[i]`` an avg-price adapter
    (``mean_clv``/``clv_cents`` read the float ``avg_price_cents`` ATTRIBUTE — never the rounded
    ``price``); ``snaps_per_fill[i]`` the closing-window snapshots; ``sides[i]`` the fill side. Fails
    loud (RESEARCH Code Example 2) on a malformed fill.
    """
    from weatherquant.db import queries
    from weatherquant.market import clv as clv_mod
    from weatherquant.price.buckets import integers_in_bucket
    from weatherquant.price.ticker import parse_ticker
    from weatherquant.registry import get_city
    from weatherquant.verify.metrics import _fill_avg_price_cents

    settled_yes: list[bool] = []
    clv_fills: list[_AvgPriceFill] = []
    snaps_per_fill: list[list] = []
    sides: list[str] = []
    for fill in window_fills:
        ticker = fill["ticker"] if hasattr(fill, "__getitem__") else fill.ticker
        event_time = fill["event_time"] if hasattr(fill, "__getitem__") else fill.event_time
        day = event_time.date()
        lo_i, hi_i, open_lo, open_hi = parse_ticker(ticker)
        lo_edge, hi_edge = integers_in_bucket(lo_i, hi_i, open_lo=open_lo, open_hi=open_hi)
        obs = queries.latest(
            bind, "observations",
            where={"city": city_key, "target_date": day, "source": "asos"},
        )
        y = next((row["daily_high_f"] for row in obs if row["daily_high_f"] is not None), None)
        if y is None:
            raise ValueError(
                f"_settle_window_fills: no settled daily_high_f for {city_key} {day} — "
                "a window fill cannot be settled (never fabricate a settlement)."
            )
        settled_yes.append(bool(lo_edge <= float(y) < hi_edge))
        # The CLV path reads the float avg_price_cents attribute (never the rounded price column).
        clv_fills.append(_AvgPriceFill(_fill_avg_price_cents(fill)))
        # The closing-window snapshots for this fill's ticker (volume-weighted CLV mid).
        all_snaps = queries.latest(bind, "market_snapshots", where={"ticker": ticker})
        snaps = clv_mod.closing_window_snapshots(
            [dict(s) for s in all_snaps], get_city(city_key), day
        )
        snaps_per_fill.append(snaps)
        side_raw = fill["side"] if hasattr(fill, "__getitem__") else fill.side
        sides.append("buy" if side_raw in ("yes", "buy") else "sell")
    return settled_yes, clv_fills, snaps_per_fill, sides


def _records_for_report(scored):
    """Group the scored paired records into the per-city ``{city, f, o, pit}`` report shape.

    ``f``/``o`` are the weatherquant YES probabilities and realized outcomes (the reliability-diagram
    inputs); ``pit`` reuses ``f`` as the calibration-transform proxy at the bucket level so the PIT
    histogram renders per city. The report renderer strata-splits on ``city`` (per-city PNGs +
    pooled) so per-city calibration error stays visible (RESEARCH §Pitfall 4).
    """
    by_city: dict = {}
    for r in scored:
        entry = by_city.setdefault(r.city, {"city": r.city, "f": [], "o": [], "pit": []})
        entry["f"].append(float(r.wq_prob))
        entry["o"].append(int(r.o_i))
        entry["pit"].append(float(r.wq_prob))
    return list(by_city.values())
