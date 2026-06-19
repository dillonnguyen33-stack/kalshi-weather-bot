"""Idempotent ingestion CLI (D-15) — the historical backfill half of the one code path.

``weatherquant ingest`` is a THIN argparse wrapper over
:func:`weatherquant.ingest.orchestrator.ingest_range`: it validates the requested city
codes (via :func:`weatherquant.registry.get_city` — raises on unknown, ASVS V5 / T-02-17)
and dates (parsed to ``date``; a malformed value is rejected BEFORE any ingest call), then
dispatches to the backfill path with ``mode="backfill"`` so each forecast's
``available_at`` is ``cycle_init + PUBLISH_LATENCY`` — never the wall clock (D-09). Because
the orchestrator routes every write through 02-02's skip-before-insert idempotency,
re-running the same range is a NO-OP (D-10).

This is the SAME orchestrator the live scheduler (:mod:`weatherquant.scheduler`) calls — the
only difference is the ``mode`` argument (D-15). The CLI uses stdlib ``argparse`` (no new
``click``/``typer`` dependency) and ``asyncio.run`` to drive the async orchestrator. It is
installed as the ``weatherquant`` console script via ``[project.scripts]`` in pyproject.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Mapping
from datetime import date, datetime, timezone
from typing import Any

from weatherquant.db.engine import get_engine, get_settings
from weatherquant.ingest import orchestrator
from weatherquant.market import clv
from weatherquant.market.auth import KalshiSigner
from weatherquant.market.client import fetch_snapshot
from weatherquant.market.persist import persist_fill, persist_snapshot
from weatherquant.registry import CITIES, get_city

logger = logging.getLogger(__name__)

# Paper snapshot persist cadence (D-03/D-04). The run_paper loop persists a market_snapshot at
# most this often (debounced) plus on any material book move. It MUST stay strictly finer than
# the CLV closing window (clv.CLV_WINDOW_MINUTES) so that when a book change falls inside the
# closing window the window holds >= 1 persisted snapshot — keeping Phase-6 CLV DERIVABLE, not
# silently sparse (PAP-04 cadence sufficiency, threat T-05-20). Asserted at run_paper start.
PAPER_SNAPSHOT_CADENCE_SECONDS = 60

# The model/source labels the CLI can ingest — the GRIB models plus the supplementary
# sources, single-sourced from the orchestrator so the CLI never drifts from the one path.
ALL_MODELS: tuple[str, ...] = (
    *orchestrator.GRIB_MODELS,
    *orchestrator.SUPPLEMENTARY_SOURCES,
)


def _parse_date(value: str) -> date:
    """Parse a ``YYYY-MM-DD`` string to a ``date``, rejecting malformed input (ASVS V5).

    Used as an argparse ``type=`` so a bad date raises ``argparse.ArgumentTypeError`` and the
    parser rejects it BEFORE any ingest call — never a silently-wrong window.
    """
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r}: expected YYYY-MM-DD ({exc})"
        ) from exc


def _validate_city(value: str) -> str:
    """Validate a city code via :func:`get_city` (raises on unknown — ASVS V5 / T-02-17)."""
    get_city(value)  # raises KeyError on an unknown code; surfaced as an arg error below.
    return value


def _city_type(value: str) -> str:
    """argparse ``type=`` wrapper turning an unknown city KeyError into an arg error."""
    try:
        return _validate_city(value)
    except KeyError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the ``weatherquant`` argparse parser (the ``ingest`` subcommand)."""
    parser = argparse.ArgumentParser(
        prog="weatherquant",
        description="Idempotent weather-model ingestion (backfill a date or range).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser(
        "ingest",
        help="Backfill forecasts/observations for a date or date range (idempotent).",
    )

    model_group = ingest.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        choices=ALL_MODELS,
        help="A single model/source to ingest (e.g. hrrr, gfs, nws, openmeteo).",
    )
    model_group.add_argument(
        "--all-models",
        action="store_true",
        help="Ingest every GRIB model and supplementary source.",
    )

    city_group = ingest.add_mutually_exclusive_group(required=True)
    city_group.add_argument(
        "--city",
        type=_city_type,
        help=f"A single Kalshi city code (one of: {', '.join(sorted(CITIES))}).",
    )
    city_group.add_argument(
        "--all-cities",
        action="store_true",
        help="Ingest every registry city.",
    )

    date_group = ingest.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--date",
        type=_parse_date,
        help="A single LST settlement date (YYYY-MM-DD).",
    )
    date_group.add_argument(
        "--start",
        type=_parse_date,
        help="Start of an inclusive date range (YYYY-MM-DD); requires --end.",
    )
    ingest.add_argument(
        "--end",
        type=_parse_date,
        help="End of an inclusive date range (YYYY-MM-DD); used with --start.",
    )

    ingest.add_argument(
        "--lead",
        type=int,
        default=0,
        help="Forecast lead hours for the GRIB models (default 0).",
    )
    ingest.add_argument(
        "--cycle-hours",
        type=str,
        default=None,
        help="Comma-separated UTC cycle init hours per day (default 0).",
    )

    # --- calibrate: fit + persist EMOS/NGR params per (city, model, lead, month) (D-13) -----
    # Mirrors the ingest selectors so the operator can calibrate every model label generically
    # (D-01 — one code path over all NOAA + supplementary labels). The same _city_type /
    # ALL_MODELS validators reject unknown cities/models BEFORE any DB call (ASVS V5 / T-03-05).
    calibrate = sub.add_parser(
        "calibrate",
        help="Fit + persist append-only EMOS/NGR calibration params per stratum.",
    )

    cal_model_group = calibrate.add_mutually_exclusive_group(required=True)
    cal_model_group.add_argument(
        "--model",
        choices=ALL_MODELS,
        help="A single model/source to calibrate (e.g. hrrr, gfs, nws, openmeteo).",
    )
    cal_model_group.add_argument(
        "--all-models",
        action="store_true",
        help="Calibrate every GRIB model and supplementary source.",
    )

    cal_city_group = calibrate.add_mutually_exclusive_group(required=True)
    cal_city_group.add_argument(
        "--city",
        type=_city_type,
        help=f"A single Kalshi city code (one of: {', '.join(sorted(CITIES))}).",
    )
    cal_city_group.add_argument(
        "--all-cities",
        action="store_true",
        help="Calibrate every registry city.",
    )

    calibrate.add_argument(
        "--lead",
        type=int,
        default=0,
        help="Forecast lead hours to calibrate (default 0).",
    )
    calibrate.add_argument(
        "--oos-fraction",
        type=float,
        default=0.3,
        help="Fraction of date-sorted samples held out for OOS validation (default 0.3).",
    )

    # --- price: OPTIONAL smoke command — latest→predict→blend→bucket/EV/Kelly (D-15/D-16) ----
    # The I/O-edge orchestration analog of run_calibrate: it pulls the latest calibration
    # params / forecasts / AFD flag, blends, and prints bucket probs / EV / Kelly stake with
    # the MARKET MIDPOINT MOCKED (D-16 — Phase 4 does NO market fetch; Phase 5 owns the book).
    # All pure math stays in weatherquant.price; the CLI is the edge only. The same _city_type
    # validator rejects an unknown city BEFORE any DB read (ASVS V5 / T-04-15).
    price = sub.add_parser(
        "price",
        help="Smoke-price a (city, date): blend latest calibration + forecasts, print "
        "bucket probs / EV / Kelly (market midpoint mocked, no market fetch).",
    )
    price.add_argument(
        "--city",
        type=_city_type,
        required=True,
        help=f"A single Kalshi city code (one of: {', '.join(sorted(CITIES))}).",
    )
    price.add_argument(
        "--date",
        type=_parse_date,
        required=True,
        help="The LST settlement date to price (YYYY-MM-DD).",
    )
    price.add_argument(
        "--lead",
        type=int,
        default=0,
        help="Forecast lead hours to price (default 0).",
    )
    price.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="A single KXHIGH range ticker to price (e.g. KXHIGHNY-62-63); optional.",
    )
    price.add_argument(
        "--market-mid",
        type=float,
        default=0.5,
        help="MOCKED market midpoint price in [0,1] (D-16 — no live market fetch; Phase 5 "
        "supplies the real (best_bid+best_ask)/2). Default 0.5.",
    )

    # --- paper: the Phase-5 loop closer — REAL live-book midpoint into the money path --------
    # Mirrors `price` (same _city_type ASVS-V5 validation, --date/--lead/--ticker) but the KEY
    # change is that run_paper supplies the REAL reflection-derived live-book midpoint (vs
    # run_price's mocked --market-mid) and feeds it into price.p_used/bucket_ev/stake_fraction
    # — closing the Phase-4 D-08/D-16 loop. It places NO real order (paper only — run_paper runs
    # only when execution_mode is NOT 'live') and persists the snapshot + (possibly partial) fill
    # via the audited market.persist path, stamped with the real WS event time.
    paper = sub.add_parser(
        "paper",
        help="Paper-trade a (city, date): feed the REAL live-book midpoint into the "
        "blend/EV/Kelly money path and simulate a fill (no real order).",
    )
    paper.add_argument(
        "--city",
        type=_city_type,
        required=True,
        help=f"A single Kalshi city code (one of: {', '.join(sorted(CITIES))}).",
    )
    paper.add_argument(
        "--date",
        type=_parse_date,
        required=True,
        help="The LST settlement date to paper-trade (YYYY-MM-DD).",
    )
    paper.add_argument(
        "--lead",
        type=int,
        default=0,
        help="Forecast lead hours to price (default 0).",
    )
    paper.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="The KXHIGH range ticker to paper-trade (e.g. KXHIGHNY-62-63).",
    )
    paper.add_argument(
        "--demo",
        action="store_true",
        help="Use the fixed Kalshi DEMO environment (REST + future WS hosts, SSRF-safe consts).",
    )
    return parser


def _resolve_range(args: argparse.Namespace) -> tuple[date, date]:
    """Resolve the inclusive ``(start, end)`` date range from the parsed args.

    ``--date`` collapses to a single-day range; ``--start`` requires ``--end`` and rejects an
    end-before-start range — all validation happens BEFORE any ingest call (ASVS V5).
    """
    if args.date is not None:
        return args.date, args.date
    if args.start is None or args.end is None:
        raise SystemExit("ingest: --start requires --end (or use --date for one day)")
    if args.end < args.start:
        raise SystemExit(f"ingest: --end {args.end} is before --start {args.start}")
    return args.start, args.end


def _resolve_models(args: argparse.Namespace) -> list[str]:
    return list(ALL_MODELS) if args.all_models else [args.model]


def _resolve_cities(args: argparse.Namespace) -> list[str]:
    return sorted(CITIES) if args.all_cities else [args.city]


def _resolve_cycle_hours(args: argparse.Namespace) -> list[int] | None:
    if not args.cycle_hours:
        return None
    try:
        return [int(h) for h in args.cycle_hours.split(",") if h.strip() != ""]
    except ValueError as exc:
        raise SystemExit(f"ingest: --cycle-hours must be comma-separated integers ({exc})")


def run_ingest(args: argparse.Namespace) -> dict[str, int]:
    """Validate args and dispatch to the orchestrator backfill path (mode=backfill, D-15).

    Cities and dates are validated (the argparse ``type=`` validators already rejected
    unknown cities and malformed dates); the engine bind is built from the validated
    ``DATABASE_URL`` and the async :func:`ingest_range` is driven via ``asyncio.run``.
    Re-running the same range is a no-op (idempotency, D-10).
    """
    start, end = _resolve_range(args)
    models = _resolve_models(args)
    cities = _resolve_cities(args)
    cycle_hours = _resolve_cycle_hours(args)

    bind = get_engine()
    logger.info(
        "ingest backfill models=%s cities=%s range=%s..%s lead=%s cycle_hours=%s",
        models,
        cities,
        start,
        end,
        args.lead,
        cycle_hours,
    )
    totals = asyncio.run(
        orchestrator.ingest_range(
            bind,
            models,
            cities,
            start,
            end,
            mode="backfill",
            lead=args.lead,
            cycle_hours=cycle_hours,
        )
    )
    logger.info("ingest complete: rows inserted per model=%s", totals)
    return totals


def run_calibrate(args: argparse.Namespace) -> dict[str, int]:
    """Validate args and run assemble→fit→evaluate→persist per stratum (D-13 / CAL-01/03).

    For each validated ``(city, model)`` pair this:

    1. assembles training pairs from the ledger (``strata.assemble_training_pairs`` — full-key
       read, K→°F seam, member aggregation), scoped to the requested ``--lead``;
    2. groups the pairs into ``(city, model, lead, month)`` strata;
    3. fits each stratum (``strata.fit_stratum_pooled`` — the pooling/shrinkage ladder records
       ``pool_level``);
    4. computes the OOS audit metrics on a TEMPORAL split over the strata's real target dates
       (``evaluate.evaluate_stratum_oos_aggregated``) when the stratum has >= 2 distinct dates
       (else leaves the OOS metrics as NaN — absence is absence, never a leaky/fabricated number);
    5. persists each stratum APPEND-ONLY via ``persist.store_calibration_params`` with
       ``available_at = now`` (the training-run completion instant, D-13) and ``trained_through``
       = the latest ``target_date`` the persisted full-stratum fit trained through.

    Cities/models were already validated by the argparse ``type=``/``choices=`` validators
    (unknown city / unknown model rejected BEFORE this runs — ASVS V5 / T-03-05). Calibration is
    synchronous (no ``asyncio.run`` — the fit is pure NumPy over in-memory arrays).

    Returns:
        ``{f"{city}:{model}": n_strata_persisted}`` — the per-pair count of persisted strata.
    """
    # Imported lazily so the ingest path (and `--help`) never pays the calibration imports.
    import math

    from weatherquant.calibrate import crps, evaluate, link, persist, strata

    models = _resolve_models(args)
    cities = _resolve_cities(args)
    lead = args.lead
    oos_fraction = args.oos_fraction

    bind = get_engine()
    available_at = datetime.now(timezone.utc)  # training-run completion instant (D-13)
    persisted: dict[str, int] = {}

    for city in cities:
        for model in models:
            pairs = [
                p
                for p in strata.assemble_training_pairs(bind, city=city, model=model)
                if p.lead == lead
            ]

            count = 0
            # Each month stratum is fit pooled toward its season parent: the n<N_MIN parent
            # fallback and the shrinkage blend fire on the production path, and a sparse-season
            # month is skipped rather than persisted as a degenerate fit.
            for month_samples, target_dates, fit in strata.fit_pooled_month_strata(
                pairs, city=city, model=model, lead=lead
            ):
                m, s2, y = month_samples.m, month_samples.s2, month_samples.y

                # crps_train must describe the PERSISTED (pooled) params: their in-sample mean
                # CRPS over the month's own rows. Taking it from the OOS train-slice fit instead
                # would store a metric for a different fit on ~0.7·N different samples.
                mu_is, sig_is = link.predict(
                    (fit.a, fit.b, fit.c, fit.d, fit.sigma_floor), m, s2
                )
                crps_train = float(crps.crps_norm(mu_is, sig_is, y).mean())

                # OOS audit metrics on a genuine TEMPORAL split (D-10): order the stratum's real
                # target dates and hold out the latest oos_fraction. This requires >= 2 DISTINCT
                # dates — with fewer the split would be positional (look-ahead leakage), so we
                # leave the metrics NaN (absence = absence) rather than persist a leaky number.
                # The persisted params' data cutoff is the latest target date — independent of
                # the diagnostic OOS split.
                # These two come from an UNPOOLED re-fit on the OOS train slice, so on a
                # pooled/parent-fallback row they describe a different fit than crps_train above —
                # a generic "does EMOS generalize?" diagnostic, not the persisted fit's OOS score.
                crps_oos = crps_baseline_oos = math.nan
                trained_through = max(target_dates)  # persisted fit's data cutoff (D-13)
                if len(set(target_dates)) >= 2:
                    oos = evaluate.evaluate_stratum_oos_aggregated(
                        target_dates, m, s2, y, oos_fraction=oos_fraction
                    )
                    crps_oos = oos.crps_oos
                    crps_baseline_oos = oos.crps_baseline_oos

                persist.store_calibration_params(
                    bind,
                    city=fit.city,
                    model=fit.model,
                    lead=fit.lead,
                    month=fit.month,
                    mean_intercept=fit.a,
                    mean_slope=fit.b,
                    var_intercept=fit.c,
                    var_slope=fit.d,
                    sigma_floor=fit.sigma_floor,
                    n_train=fit.n_train,
                    pool_level=fit.pool_level,
                    crps_train=crps_train,
                    crps_oos=crps_oos,
                    crps_baseline_oos=crps_baseline_oos,
                    trained_through=trained_through,
                    available_at=available_at,
                )
                count += 1

            persisted[f"{city}:{model}"] = count
            logger.info("calibrate city=%s model=%s lead=%s strata=%d", city, model, lead, count)

    return persisted


def _blend_distribution(
    bind: Any, city: str, target: date, lead: int
) -> dict[str, Any]:
    """Shared latest→predict→blend→sufficiency body (reused by run_price AND run_paper).

    Reads the latest calibration params + forecast members for ``(city, target, lead)``,
    aggregates members to ``(mean_f, var_f)`` at the ONE K→°F seam, reconstructs ``(μ_i, σ_i)``
    via ``link.predict``, computes accuracy weights from ``crps_oos`` and Vincentizes to
    ``(μ_blend, σ_blend)``, reads the AFD disagreement flag, and resolves the conservative
    (smallest-sufficiency-ramp) representative ``n_train``/``pool_level`` — the conservative
    pick keeps the sized stake honest when strata disagree on sufficiency. Fails loud on a NULL
    ``n_train`` and on no usable model. Pure money-path math stays in ``weatherquant.price``;
    this is the shared DB-read edge.

    Returns a dict with ``used_models``/``mu_blend``/``sigma_blend``/``afd_flag``/``n_train``/
    ``pool_level``/``cap`` — everything the bucket/EV/Kelly leg needs, market-midpoint-agnostic.
    """
    import numpy as np

    from weatherquant import price as pricing
    from weatherquant.calibrate import link, strata
    from weatherquant.db import queries

    cap = get_settings().max_position_fraction
    month = target.month
    forecasts = queries.latest(
        bind, "forecasts", where={"city": city, "target_date": target, "lead": lead}
    )
    cal_rows = queries.latest(
        bind, "calibration_params", where={"city": city, "lead": lead, "month": month}
    )
    cal_by_model = {r["model"]: r for r in cal_rows}

    members_by_model: dict[str, list[float]] = {}
    for row in forecasts:
        if row["temp_kelvin"] is None:
            continue
        members_by_model.setdefault(row["model"], []).append(
            strata.kelvin_to_fahrenheit(float(row["temp_kelvin"]))
        )

    mus: list[float] = []
    sigmas: list[float] = []
    crps_oos: list[float] = []
    used_models: list[str] = []
    for model, vals in members_by_model.items():
        cal = cal_by_model.get(model)
        if cal is None or not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        mean_f = float(arr.mean())
        var_f = float(arr.var()) if arr.size > 1 else 0.0
        params = (
            cal["mean_intercept"],
            cal["mean_slope"],
            cal["var_intercept"],
            cal["var_slope"],
            cal["sigma_floor"],
        )
        if any(v is None for v in params):
            continue
        mu_i, sigma_i = link.predict(params, np.array([mean_f]), np.array([var_f]))
        mus.append(float(mu_i[0]))
        sigmas.append(float(sigma_i[0]))
        crps_oos.append(
            float(cal["crps_oos"]) if cal["crps_oos"] is not None else float("nan")
        )
        used_models.append(model)

    if not used_models:
        raise SystemExit(
            f"no model has BOTH latest forecasts and calibration for "
            f"city={city} date={target} lead={lead} month={month}."
        )

    weights = pricing.accuracy_weights(np.asarray(crps_oos, dtype=np.float64))
    mu_blend, sigma_blend = pricing.blend_gaussians(
        np.asarray(mus), np.asarray(sigmas), weights
    )

    afd_rows = queries.latest(
        bind, "observations", where={"city": city, "target_date": target, "source": "afd"}
    )
    afd_flag = bool(afd_rows)

    def _suff_for(model: str) -> tuple[float, str]:
        cal = cal_by_model[model]
        raw_n = cal["n_train"]
        if raw_n is None:
            raise SystemExit(
                f"calibration row for model={model} has NULL n_train — "
                f"cannot size (would silently zero the stake)."
            )
        pool = str(cal["pool_level"] or "")
        return pricing.sufficiency_ramp(int(raw_n), pool), model

    _, rep_model = min(_suff_for(m) for m in used_models)
    rep_cal = cal_by_model[rep_model]

    return {
        "used_models": used_models,
        "mu_blend": mu_blend,
        "sigma_blend": sigma_blend,
        "afd_flag": afd_flag,
        "n_train": int(rep_cal["n_train"]),
        "pool_level": str(rep_cal["pool_level"] or ""),
        "cap": cap,
    }


def run_price(args: argparse.Namespace) -> dict[str, Any]:
    """Smoke-price one (city, date): latest→predict→blend→bucket/EV/Kelly (D-15/D-16).

    The I/O-edge orchestration analog of :func:`run_calibrate`. For the validated
    ``(city, date, lead)`` this:

    1. reads the latest ``calibration_params`` per model and the latest ``forecasts`` members
       for the target date via ``queries.latest`` (full natural key — no under-specified read);
    2. aggregates each model's members to ``(mean_f, var_f)`` in °F using
       ``strata.kelvin_to_fahrenheit`` at the boundary (D-15 — the ONE K→°F seam, never
       re-derived) and reconstructs ``(μ_i, σ_i)`` via ``link.predict`` reused verbatim;
    3. computes accuracy weights from the persisted ``crps_oos`` (``price.accuracy_weights``)
       and blends to ``(μ_blend, σ_blend)`` via Vincentization (``price.blend_gaussians``);
    4. reads the AFD disagreement flag from the latest ``observations`` ``source='afd'`` row;
    5. prices the requested bucket(s): bucket probability (``price.bucket_prob`` after
       ``price.parse_ticker`` + ``price.integers_in_bucket``), fee-corrected EV
       (``price.bucket_ev``) and the capped fractional-Kelly stake (``price.stake_fraction``)
       — all with the MARKET MIDPOINT MOCKED from ``--market-mid`` (D-16: no market fetch).

    The city was already validated by the argparse ``type=_city_type`` (unknown city rejected
    BEFORE this runs — ASVS V5 / T-04-15). All pure math lives in :mod:`weatherquant.price`;
    this function is the offline-untestable I/O edge only and returns a small result dict.
    """
    # Imported lazily so the ingest/calibrate paths (and `--help`) never pay these imports.
    from weatherquant import price as pricing

    city = args.city
    target = args.date
    lead = args.lead
    market_mid = args.market_mid
    if not (0.0 <= market_mid <= 1.0):  # mocked midpoint still validated (ASVS V5)
        raise SystemExit(f"price: --market-mid must be in [0, 1], got {market_mid}")

    bind = get_engine()
    # The shared latest→predict→blend→sufficiency body (reused by run_paper); the ONLY
    # difference there is the REAL live-book midpoint replacing this MOCKED --market-mid (D-16).
    blend = _blend_distribution(bind, city, target, lead)
    used_models = blend["used_models"]
    mu_blend = blend["mu_blend"]
    sigma_blend = blend["sigma_blend"]
    afd_flag = blend["afd_flag"]
    n_train = blend["n_train"]
    pool_level = blend["pool_level"]
    cap = blend["cap"]

    logger.info(
        "price city=%s date=%s lead=%s models=%s mu_blend=%.2f sigma_blend=%.2f afd=%s",
        city, target, lead, used_models, mu_blend, sigma_blend, afd_flag,
    )

    result: dict[str, Any] = {
        "city": city,
        "date": target.isoformat(),
        "lead": lead,
        "models": used_models,
        "mu_blend": mu_blend,
        "sigma_blend": sigma_blend,
        "afd_flag": afd_flag,
        "market_mid": market_mid,
        "buckets": [],
    }

    if args.ticker is not None:
        lo, hi, open_lo, open_hi = pricing.parse_ticker(args.ticker)
        c_lo, c_hi = pricing.integers_in_bucket(lo, hi, open_lo, open_hi)
        prob = pricing.bucket_prob(
            mu_blend, sigma_blend, c_lo, c_hi, open_lo, open_hi
        )
        # EV and the sized stake MUST share one decision basis. bucket_ev shrinks the model prob
        # toward the market mid internally (D-08, p_used); size Kelly on that SAME shrunk belief
        # so the printed edge and the stake agree in sign near the boundary.
        pu = pricing.p_used(prob, market_mid)
        ev = pricing.bucket_ev(prob, market_mid, market_mid)
        stake = pricing.stake_fraction(
            pu, market_mid, pricing.exact_fee(1, market_mid),
            sigma_blend, n_train, pool_level, afd_flag, cap=cap,
        )
        bucket = {
            "ticker": args.ticker, "prob": prob, "ev": ev, "stake_fraction": stake,
        }
        result["buckets"].append(bucket)
        print(
            f"price {city} {target} {args.ticker}: P={prob:.4f} EV={ev:+.4f} "
            f"stake={stake:.4f} (mocked mid={market_mid})"
        )
    else:
        print(
            f"price {city} {target} lead={lead}: blend N(mu={mu_blend:.2f}, "
            f"sigma={sigma_blend:.2f}) over models={used_models} afd={afd_flag} "
            f"(no --ticker → distribution only; market mid mocked={market_mid})"
        )

    return result


# The minimum Kelly stake fraction below which the EV+stake gate declines to place a paper
# order (D-04). One sized position per (market, side) held to settlement — a sub-minimum stake
# is "no edge worth the spread", not a churned micro-order.
PAPER_MIN_STAKE_FRACTION = 1e-4


def _reflection_midpoint_cents(book: object) -> float:
    """Derive the live-book midpoint in FLOAT-VALUED CENTS from the REFLECTED best levels.

    Kalshi quotes only bids; the yes ask is reflected as ``100 - best_no_bid`` (cents) via
    :func:`weatherquant.market.reflect.yes_ask_levels` (best/cheapest first). The midpoint is
    ``(best_yes_bid + best_yes_ask) / 2`` in CENTS (the half-cent midpoint, e.g. 50.5 for a
    yes bid 50 / reflected yes ask 51). This is the value PERSISTED as ``market_snapshots.mid``
    — unit-consistent with ``best_yes_bid``/``best_no_bid`` (integer cents) and the fill's
    ``avg_price_cents``, so ``clv.clv_cents`` subtracts with NO conversion. The [0,1] pricing
    value is this midpoint divided by 100 — ``run_paper`` computes it inline as
    ``mid_unit = mid_cents / 100.0`` (3 chars of clear code, no wrapper helper).

    Fails loud (raise) when either side of the book is empty — no two-sided market → no
    derivable midpoint (absence = absence, never a fabricated mid).
    """
    from weatherquant.market import reflect

    # The yes BID side comes straight off the book; the yes ASK is reflected from the no bids
    # via the ONE reflection seam (yes_ask = 100 - no_bid). Never read a native ask (there is
    # none) and never re-derive the 100 - price reflection outside reflect.py.
    yes_raw = book["yes"] if isinstance(book, Mapping) else getattr(book, "yes")
    yes_bid_prices = [int(price) for price, _ in yes_raw]
    yes_asks = reflect.yes_ask_levels(book)  # reflected from the no bids, cheapest first
    if not yes_bid_prices or not yes_asks:
        raise SystemExit(
            "paper: book is one-sided (missing a yes bid or a reflected yes ask) — "
            "cannot derive a two-sided midpoint (no fabricated mid)."
        )
    best_yes_bid = max(yes_bid_prices)
    best_yes_ask = yes_asks[0][0]  # cheapest reflected yes ask = 100 - best_no_bid
    # Reject a crossed/locked reflected book (best bid >= best ask) at the money-path edge
    # (CR-01). A crossed book lets the reflected ask "prices" violate cheapest-first, so the
    # taker sweep would credit an average that corresponds to no walk a real taker could
    # execute; the downstream 1..99c rounded-band guard can still pass on a crossed book while
    # the credited price is wrong. Fail loud rather than feed a fabricated mid/fill into the
    # ledger and CLV (absence/inconsistency is not a tradeable two-sided market).
    if best_yes_bid >= best_yes_ask:
        raise SystemExit(
            f"paper: reflected book is crossed/locked (best_yes_bid={best_yes_bid}c >= "
            f"best_yes_ask={best_yes_ask}c) — refusing to derive a midpoint or credit a fill "
            "from a non-tradeable book (CR-01)."
        )
    return (best_yes_bid + best_yes_ask) / 2.0


def _snapshot_event_time(snapshot: Mapping[str, Any]) -> datetime:
    """Thin CLI wrapper over the single ``clv.snapshot_event_time`` seam (DD-1, D-08).

    The snapshot event-time parse has ONE home (``clv.snapshot_event_time``) — it already
    handles the full key set (``event_time``/``available_at``/``snapshot_for``), so the cli no
    longer drifts on ``available_at``. This wrapper only translates the ``ValueError`` (fail
    loud, never now()) into the CLI's ``SystemExit`` surface; it duplicates no parse body. The
    persisted ``available_at`` is therefore the real observed instant of the book, never the
    wall clock (back-dating destroys Phase-6 no-look-ahead, D-08).
    """
    try:
        return clv.snapshot_event_time(snapshot)
    except ValueError as exc:
        raise SystemExit(f"paper: {exc} — refusing to stamp with now() (D-08).") from exc


def run_paper(args: argparse.Namespace) -> dict[str, Any]:
    """Paper-trade one (city, date, ticker): REAL live-book midpoint into the money path (D-08/D-16).

    The Phase-5 loop closer. For the validated ``(city, date, lead, ticker)`` this:

    1. confirms paper mode — there is NO order-submission path anywhere under ``market/`` this
       milestone, so an accidental live order is structurally unreachable (D-15, T-05-14);
    2. pulls a REAL Kalshi orderbook snapshot via :func:`weatherquant.market.client.fetch_snapshot`
       (signed REST; demo host with ``--demo``);
    3. derives the live-book midpoint from the REFLECTED best levels
       (``yes_ask = 100 - best_no_bid`` → ``[0,1]``, the ONE reflection seam) — this is the
       value the loop closes on;
    4. reuses the shared :func:`_blend_distribution` body, then feeds the REAL midpoint (not a
       mock) into ``price.p_used`` → ``price.bucket_ev`` → ``price.stake_fraction`` — closing
       the Phase-4 D-08/D-16 loop;
    5. applies the positive-EV + minimum-stake gate (D-04): one sized position per (market,
       side), held to settlement; a sub-minimum or non-positive-EV intent simulates NO fill;
    6. on a positive intent, simulates a (possibly partial) fill via ``fills.taker_sweep``
       against the reflected asks;
    7. PERSISTS exactly ONE market snapshot per invocation and (on a positive intent) the fill
       via ``market.persist`` — both stamped with the REAL WS event time (D-08). NOTE: this
       command is single-shot; there is NO in-process debounced cadence loop here. Whether the
       CLV closing window holds >= 1 persisted snapshot (PAP-04 cadence sufficiency) is the
       OPERATOR's per-invocation responsibility — run ``paper`` often enough during the closing
       window. ``PAPER_SNAPSHOT_CADENCE_SECONDS`` is the design-time TARGET cadence a future
       feed-driven loop must honour, not a guarantee this single-shot command provides;
    8. places NO real order. Returns a small result dict (``midpoint``/``p_used``/``ev``/
       ``stake``/fill summary/persisted-snapshot event times).

    All pure math stays in :mod:`weatherquant.price` / :mod:`weatherquant.market.fills`; this
    is the I/O edge.
    """
    from weatherquant import price as pricing
    from weatherquant.market import fills, reflect

    # Design-time invariant on the two cadence constants (PAP-04 cadence sufficiency, T-05-20):
    # the TARGET snapshot cadence must stay strictly finer than the CLV closing window so a
    # future feed-driven loop honouring PAPER_SNAPSHOT_CADENCE_SECONDS keeps the window dense.
    # This single-shot command persists ONE snapshot per invocation and runs no cadence loop, so
    # actual window density is the operator's per-invocation responsibility (WR-02) — but the
    # constant relationship still must hold for the loop this targets. Raise (not assert) so the
    # check survives `python -O`/PYTHONOPTIMIZE, mirroring the writer's documented discipline
    # (IN-05).
    if PAPER_SNAPSHOT_CADENCE_SECONDS >= clv.CLV_WINDOW_MINUTES * 60:
        raise RuntimeError(
            "PAPER_SNAPSHOT_CADENCE_SECONDS must be strictly finer than the CLV closing window "
            "(clv.CLV_WINDOW_MINUTES * 60) so a feed-driven cadence loop never leaves the window "
            "silently sparse (PAP-04, T-05-20)."
        )

    city = args.city
    target = args.date
    lead = args.lead
    ticker = args.ticker

    settings = get_settings()
    # The SIMULATOR-only gate: run_paper runs ONLY when execution_mode is NOT 'live' — in live
    # mode the real order flow (Gate 2) would run, not the shadow simulator, so the simulator
    # bows out. This is a distinct check from market.fills.assert_paper_mode, which is the
    # separate, inverse (fail-CLOSED) fence guarding the future order path; it is NOT the check
    # applied here.
    if settings.execution_mode == "live":
        raise SystemExit(
            "paper: execution_mode='live' — the paper-fill simulator does not run in live "
            "mode (no order-submission path exists this milestone, D-15/T-05-14)."
        )

    bind = get_engine()
    signer = KalshiSigner.from_settings(settings)

    import httpx

    from weatherquant.market.client import _resolve_hosts

    # Resolve the COMPLETE (ws_url, rest_host) environment pair from the single --demo flag via
    # the one host-resolution seam (WR-06). Today run_paper does a one-shot REST fetch with no WS,
    # so only rest_host is consumed here — but resolving BOTH from --demo through _resolve_hosts
    # keeps the flag's meaning complete and stable: when a WS feed is wired into run_paper, ws_url
    # is already the matching demo/prod host, never a cross-environment book on the money path.
    demo = bool(getattr(args, "demo", False))
    ws_url, rest_host = _resolve_hosts(demo)
    del ws_url  # not consumed until the WS feed lands; resolved here so the flag stays complete.

    async def _fetch() -> dict[str, Any]:
        async with httpx.AsyncClient() as http:
            return await fetch_snapshot(http, signer.sign, ticker, rest_host=rest_host)

    snapshot = asyncio.run(_fetch())
    event_time = _snapshot_event_time(snapshot)

    # The REAL reflection-derived live-book midpoint is kept in TWO units because persistence
    # and pricing need different ones: mid_cents (float-valued CENTS) is PERSISTED as
    # market_snapshots.mid so it is unit-consistent with best_*_bid/avg_price_cents and CLV
    # subtracts with no conversion; mid_unit = mid_cents/100.0 is the [0,1] value pricing needs
    # because p_used/EV/Kelly operate on probabilities (it closes the Phase-4 D-08/D-16 loop).
    mid_cents = _reflection_midpoint_cents(snapshot)
    mid_unit = mid_cents / 100.0
    if not (0.0 <= mid_unit <= 1.0):  # the reflected mid must be a valid probability (ASVS V5)
        raise SystemExit(f"paper: reflected midpoint {mid_unit} is not in [0, 1]")

    blend = _blend_distribution(bind, city, target, lead)
    mu_blend = blend["mu_blend"]
    sigma_blend = blend["sigma_blend"]
    afd_flag = blend["afd_flag"]
    n_train = blend["n_train"]
    pool_level = blend["pool_level"]
    cap = blend["cap"]

    lo, hi, open_lo, open_hi = pricing.parse_ticker(ticker)
    c_lo, c_hi = pricing.integers_in_bucket(lo, hi, open_lo, open_hi)
    prob = pricing.bucket_prob(mu_blend, sigma_blend, c_lo, c_hi, open_lo, open_hi)

    # Feed the REAL [0,1] mid_unit into the SAME money path run_price mocks (D-08/D-16 loop
    # closed): p_used shrinks the model prob toward the real mid, EV and Kelly size on that
    # shrunk belief. mid_unit (NOT mid_cents) feeds pricing — the pricing path is in [0,1].
    pu = pricing.p_used(prob, mid_unit)
    ev = pricing.bucket_ev(prob, mid_unit, mid_unit)
    stake = pricing.stake_fraction(
        pu, mid_unit, pricing.exact_fee(1, mid_unit),
        sigma_blend, n_train, pool_level, afd_flag, cap=cap,
    )

    # Persist exactly ONE snapshot for this invocation (this command is single-shot; there is no
    # in-process cadence loop, WR-02), stamped with the REAL WS event time (D-08) and carrying the
    # load-bearing book fields + raw book JSONB (D-03). CLV closing-window density (PAP-04) is the
    # operator's per-invocation responsibility: each run lands a snapshot whose available_at is
    # the real observed instant, so running `paper` during the closing window keeps it covered.
    yes_levels = snapshot.get("yes") or []
    no_levels = snapshot.get("no") or []
    # best_*_bid are the PRICE columns: the max (best) bid price on each side (cents). The yes ASK
    # is the reflection 100 - best_no_bid (reflect.py); these prices back the persisted yes mid.
    best_yes_bid = max((int(p) for p, _ in yes_levels), default=None)
    best_no_bid = max((int(p) for p, _ in no_levels), default=None)
    # The per-snapshot volume is the liquidity BEHIND the persisted yes mid — the top-of-book
    # two-sided SUPPORTING size min(best_yes_bid_size, best_yes_ask_size). The persisted mid is the
    # yes-side midpoint (best_yes_bid + best_yes_ask)/2 (via _reflection_midpoint_cents), so the
    # size that actually supports it is the size you can trade at the touch: the smaller of the
    # best-yes-bid size and the best-yes-ask size. Because Kalshi quotes only bids and the yes ask
    # is reflected as 100 - best_no_bid carrying the best NO bid's SIZE (reflect.py),
    # best_yes_ask_size == best_no_bid_size — so this is min(best_yes_bid_size, best_no_bid_size).
    #
    # WHY this over 05-06 MD-01's sum(yes sizes)+sum(no sizes): that two-sided UNION depth
    # over-weights a snapshot deep on the OPPOSITE (no) side but thin on the yes side — its yes-mid
    # is barely supported yet would carry a large CLV weight, biasing the closing mid toward
    # opposite-side-heavy instants (CORR-MED-3). Narrowing to the supporting top-of-book size
    # reconciles 05-06 MD-01 (still a REAL off-the-feed liquidity signal, no fabrication) while
    # weighting each mid by the liquidity that genuinely backs THIS mid. The reflection's
    # 100 - price is NOT re-derived here: the supporting yes-ask size IS the best-no-bid size by
    # construction, read straight off the no side. Cast to int (whole contracts).
    best_yes_bid_size = max(yes_levels, key=lambda lvl: int(lvl[0]))[1] if yes_levels else None
    best_no_bid_size = max(no_levels, key=lambda lvl: int(lvl[0]))[1] if no_levels else None
    if best_yes_bid_size is None or best_no_bid_size is None:
        raise SystemExit(
            "paper: book is one-sided — cannot derive the top-of-book supporting size behind "
            "the persisted mid (no fabricated volume)."
        )
    volume = int(min(int(best_yes_bid_size), int(best_no_bid_size)))
    # The market_snapshots natural key is (ticker, snapshot_for). RFC-1123 Date headers carry only
    # SECOND resolution (WR-01's preferred event_time source), so two distinct book states fetched
    # in the same wall-clock second would share an ISO-only snapshot_for; queries.latest's
    # DISTINCT ON (ticker, snapshot_for) would then silently discard one of the two from the
    # closing-window mid (WR-03). Append the WS seq (the monotonic per-book sequence that DOES
    # differ between distinct states) so two same-second states remain distinct natural-key rows
    # the closing window can both weight. The closing-window axis for persisted rows is the
    # event_time/available_at datetime (preferred by clv.snapshot_event_time), which the seq
    # suffix does not touch; the parser strips the suffix before its ISO fallback.
    seq = snapshot.get("seq")
    snapshot_for = (
        f"{event_time.isoformat()}#{int(seq)}" if seq is not None else event_time.isoformat()
    )
    persisted_snapshot_times: list[str] = []
    rc_snap = persist_snapshot(
        bind,
        ticker=ticker,
        snapshot_for=snapshot_for,
        best_yes_bid=best_yes_bid,
        best_no_bid=best_no_bid,
        # Persist CENTS: mid_cents is unit-consistent with best_*_bid/avg_price_cents so CLV
        # subtracts directly. NEVER persist the [0,1] mid_unit here.
        mid=mid_cents,
        volume=volume,
        seq=snapshot.get("seq"),
        detail={"yes": snapshot.get("yes"), "no": snapshot.get("no")},
        available_at=event_time,
    )
    if rc_snap == 1:
        persisted_snapshot_times.append(event_time.isoformat())

    # The positive-EV + minimum-stake gate (D-04): only a positive, sized intent simulates a
    # paper order. One sized position per (market, side), held to settlement — no churn.
    fill_summary: dict[str, Any] | None = None
    if ev > 0.0 and stake >= PAPER_MIN_STAKE_FRACTION:
        # Size the paper order conservatively at 1 contract for the shadow sim (the Gate-1
        # credited path is the taker sweep at achievable liquidity; bankroll→contract count
        # realism is a Phase-6 concern). Real sizing lands when live sizing is enabled (Gate 2).
        want_count = 1
        yes_asks = reflect.yes_ask_levels(snapshot)
        fill = fills.taker_sweep(yes_asks, want_count, event_time=event_time)
        if fill is not None:
            trade_id = f"{ticker}:{snapshot_for}:yes"
            # FAIL LOUD on a taker price that rounds out of the valid 1..99c band. The maker-zero
            # guard in insert_fill (writer.py) only fires for is_maker is True, so a taker fill
            # whose size-weighted average rounds to 0 (a 0c reflected ask from a no_bid of 100, or
            # a malformed/edge book) would otherwise persist price=0 and corrupt CLV as
            # closing_mid - 0 — the exact failure mode the maker guard exists to prevent, on the
            # path it exempts (WR-04). Surface the out-of-band sweep here rather than fabricate a
            # free fill; survives python -O via SystemExit (not assert).
            fill_price_cents = int(round(fill.avg_price_cents))
            if not (1 <= fill_price_cents <= 99):
                raise SystemExit(
                    f"paper: taker fill on {ticker} rounds to price={fill_price_cents}c, outside "
                    f"the valid 1..99c band (avg_price_cents={fill.avg_price_cents!r}) — refusing "
                    "to persist a fabricated 0c/out-of-band fill that would corrupt CLV (WR-04)."
                )
            persist_fill(
                bind,
                ticker=ticker,
                trade_id=trade_id,
                side="yes",
                price=fill_price_cents,
                count=fill.count,
                # Fee on the ACHIEVED size-weighted fill price (in [0,1]), NOT the decision-time
                # mid — so the persisted fee is internally consistent with the persisted price
                # whenever the sweep clears away from the mid (thin/multi-level/partial). Feeing
                # on mid_unit double-counted the slippage into the audited fee (WR-03).
                fee=int(round(pricing.exact_fee(fill.count, fill.avg_price_cents / 100.0) * 100)),
                is_maker=fill.is_maker,
                event_time=fill.event_time,
                bucket_prob=prob,
                ev=ev,
                kelly_stake=stake,
                # Record the decision mid alongside the achieved price + the per-fill slippage
                # (achieved - mid) so an audit of the fills row can distinguish "edge realized"
                # from "edge eaten by the sweep" rather than infer it — the order was sized on
                # mid_cents but clears at avg_price_cents, which a multi-level partial sweep can
                # push materially worse than the touch the mid implied (WR-05).
                detail={
                    "avg_price_cents": fill.avg_price_cents,
                    "partial": fill.partial,
                    "mid_cents": mid_cents,
                    "slippage_cents": fill.avg_price_cents - mid_cents,
                },
                available_at=fill.event_time,
            )
            fill_summary = {
                "trade_id": trade_id,
                "count": fill.count,
                "avg_price_cents": fill.avg_price_cents,
                "partial": fill.partial,
                "shortfall": fill.shortfall,
            }

    logger.info(
        "paper city=%s date=%s ticker=%s midpoint=%.4f mid_cents=%.2f p_used=%.4f ev=%+.4f "
        "stake=%.4f fill=%s",
        city, target, ticker, mid_unit, mid_cents, pu, ev, stake, fill_summary,
    )
    print(
        f"paper {city} {target} {ticker}: mid={mid_unit:.4f} ({mid_cents:.2f}c) P={prob:.4f} "
        f"EV={ev:+.4f} stake={stake:.4f} "
        f"fill={'none' if fill_summary is None else fill_summary['count']}"
    )

    return {
        "city": city,
        "date": target.isoformat(),
        "lead": lead,
        "ticker": ticker,
        "models": blend["used_models"],
        # The [0,1] pricing midpoint fed into p_used/EV/Kelly; mid_cents is its persisted-unit
        # twin (FLOAT-VALUED CENTS, the unit market_snapshots.mid stores).
        "midpoint": mid_unit,
        "mid_cents": mid_cents,
        "p_used": pu,
        "prob": prob,
        "ev": ev,
        "stake": stake,
        "fill": fill_summary,
        "persisted_snapshot_times": persisted_snapshot_times,
    }


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point (``[project.scripts] weatherquant = weatherquant.cli:main``).

    Parses args, dispatches the ``ingest`` subcommand to :func:`run_ingest`, and returns a
    process exit code (0 on success). Argument validation errors (unknown city, malformed
    date) are raised by argparse BEFORE any ingest call (ASVS V5).
    """
    logging.basicConfig(level=logging.INFO)
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        totals = run_ingest(args)
        inserted = sum(totals.values())
        print(f"ingest complete: {inserted} row(s) inserted ({totals})")
        return 0
    if args.command == "calibrate":
        persisted = run_calibrate(args)
        total = sum(persisted.values())
        print(f"calibrate complete: {total} stratum row(s) persisted ({persisted})")
        return 0
    if args.command == "price":
        run_price(args)  # prints the blend/bucket/EV/Kelly smoke line(s) itself (D-16)
        return 0
    if args.command == "paper":
        run_paper(args)  # prints the real-midpoint loop-closure smoke line itself (D-08/D-16)
        return 0
    parser.error(f"unknown command: {args.command}")  # NoReturn — exits non-zero


if __name__ == "__main__":
    raise SystemExit(main())
