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
from datetime import date, datetime, timezone
from typing import Any

from weatherquant.db.engine import get_engine, get_settings
from weatherquant.ingest import orchestrator
from weatherquant.registry import CITIES, get_city

logger = logging.getLogger(__name__)

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
            # Each month stratum is fit pooled toward its season parent (CR-01 / D-08): the
            # n<N_MIN parent fallback and the shrinkage blend now fire on the production path,
            # and a sparse-season month is skipped rather than persisted as a degenerate fit.
            for month_samples, target_dates, fit in strata.fit_pooled_month_strata(
                pairs, city=city, model=model, lead=lead
            ):
                m, s2, y = month_samples.m, month_samples.s2, month_samples.y

                # crps_train must describe the PERSISTED (pooled) params: their in-sample mean
                # CRPS over the month's own rows (WR-01). Taking it from the OOS train-slice fit
                # instead would store a metric for a different fit on ~0.7·N different samples.
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
                # WR-05: these two come from an UNPOOLED re-fit on the OOS train slice, so on a
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
    import numpy as np

    from weatherquant import price as pricing
    from weatherquant.calibrate import link, strata
    from weatherquant.db import queries

    city = args.city
    target = args.date
    lead = args.lead
    market_mid = args.market_mid
    if not (0.0 <= market_mid <= 1.0):  # mocked midpoint still validated (ASVS V5)
        raise SystemExit(f"price: --market-mid must be in [0, 1], got {market_mid}")

    bind = get_engine()
    cap = get_settings().max_position_fraction  # position cap from typed config (D-13)

    month = target.month
    forecasts = queries.latest(
        bind, "forecasts", where={"city": city, "target_date": target, "lead": lead}
    )
    cal_rows = queries.latest(
        bind, "calibration_params", where={"city": city, "lead": lead, "month": month}
    )
    cal_by_model = {r["model"]: r for r in cal_rows}

    # Aggregate forecast members per model to (mean_f, var_f) in °F at the K→°F seam (D-15).
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
        if cal is None or not vals:  # no calibration for this model → it drops out (D-03)
            continue
        arr = np.asarray(vals, dtype=np.float64)
        mean_f = float(arr.mean())
        var_f = float(arr.var()) if arr.size > 1 else 0.0  # deterministic model → var 0
        params = (
            cal["mean_intercept"],
            cal["mean_slope"],
            cal["var_intercept"],
            cal["var_slope"],
            cal["sigma_floor"],
        )
        mu_i, sigma_i = link.predict(
            params, np.array([mean_f]), np.array([var_f])
        )
        mus.append(float(mu_i[0]))
        sigmas.append(float(sigma_i[0]))
        crps_oos.append(
            float(cal["crps_oos"]) if cal["crps_oos"] is not None else float("nan")
        )
        used_models.append(model)

    if not used_models:
        raise SystemExit(
            f"price: no model has BOTH latest forecasts and calibration for "
            f"city={city} date={target} lead={lead} month={month}."
        )

    weights = pricing.accuracy_weights(np.asarray(crps_oos, dtype=np.float64))
    mu_blend, sigma_blend = pricing.blend_gaussians(
        np.asarray(mus), np.asarray(sigmas), weights
    )

    # AFD disagreement flag from the latest observations source='afd' row (soft, D-12).
    afd_rows = queries.latest(
        bind, "observations", where={"city": city, "target_date": target, "source": "afd"}
    )
    afd_flag = bool(afd_rows)

    # Resolve the calibration n_train/pool_level for sufficiency (use the highest-weight model
    # actually used — a representative of the blend's data sufficiency).
    lead_cal = cal_by_model[used_models[int(np.argmax(weights))]]
    n_train = int(lead_cal["n_train"] or 0)
    pool_level = str(lead_cal["pool_level"] or "")

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
        ev = pricing.bucket_ev(prob, market_mid, market_mid)
        stake = pricing.stake_fraction(
            prob, market_mid, pricing.exact_fee(1, market_mid),
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
    parser.error(f"unknown command: {args.command}")
    return 2  # unreachable — parser.error exits


if __name__ == "__main__":
    raise SystemExit(main())
