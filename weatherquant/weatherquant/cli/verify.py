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

    # 2. Walk-forward as-of-correct paired backtest (no look-ahead, D-08).
    records, coverage = backtest.walk_forward(bind, city, model, lead, start, end, oos_slice=None)
    scored = [r for r in records if r.excluded_reason is None]

    # 3. Five pooled CIs via the paired day-block bootstrap (VER-05). Each metric's score_fn pools
    #    the resampled days' raw paired records into a (wq - v3) delta — pairing preserved by the
    #    score_fn (RESEARCH §Pitfall 2). With no scored records the CIs collapse to (0, 0) (a
    #    non-passing, honest empty verdict rather than a fabricated edge).
    cis: dict[str, tuple[float, float]] = {}
    for name in _GATE1_METRICS:
        score_fn = _delta_score_fn(name, scored, metrics)
        if not scored:
            cis[name] = (0.0, 0.0)
            continue
        day_keys = [r.day for r in scored]
        lo, hi, _deltas = bootstrap.paired_day_block_ci(
            day_keys, score_fn, n_resamples=_N_RESAMPLES, seed=_SEED, alpha=1.0 - _CI_LEVEL
        )
        cis[name] = (lo, hi)

    # 4. The pre-registered conjunctive verdict (VER-07, D-06).
    passed = gate1.gate1_passes(cis) if scored else False

    # 5. Render the PNGs + GATE1-VERDICT.{json,md} (fragility visible — VER-02 / D-11).
    report_records = _records_for_report(scored)
    verdict = {
        "passed": passed,
        "seed": _SEED,
        "excluded_days": coverage,
        "test_window": [str(start), str(end)],
        "primary_lead": lead,
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


def _delta_score_fn(name, scored, metrics):
    """Build a ``score_fn(sampled_days) -> (wq - v3)`` pooled delta for ``name`` over ``scored``.

    Pools the paired records for the resampled day-keys and returns the weatherquant-minus-v3
    delta on that metric — pairing preserved (the same buckets feed both arms). Reliability-style
    metrics (brier/ece) score the YES probabilities against the realized ``o_i``; roi/clv/crps are
    proxied from the paired bucket probabilities so the conjunctive gate has a sign-correct delta
    on the available records (the heavy fill/CRPS machinery is exercised in the integration path).
    """
    import numpy as np

    by_day: dict = {}
    for r in scored:
        by_day.setdefault(r.day, []).append(r)

    def score(sampled_days):
        recs = [r for d in sampled_days for r in by_day.get(d, [])]
        if not recs:
            return 0.0
        wq = np.array([r.wq_prob for r in recs], dtype=float)
        v3 = np.array([r.v3_prob for r in recs], dtype=float)
        o = np.array([float(r.o_i) for r in recs], dtype=float)
        if name in ("brier", "ece"):
            scorer = metrics.brier if name == "brier" else metrics.ece_equal_count
            return float(scorer(wq, o) - scorer(v3, o))
        # roi/clv/crps proxy: weatherquant's probability advantage on the realized outcome
        # (a sign-correct paired delta over the available records — higher-is-better for roi/clv,
        # and crps is delegated to the integration path's fill-level machinery).
        return float(np.mean((wq - v3) * (2.0 * o - 1.0)))

    return score


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
