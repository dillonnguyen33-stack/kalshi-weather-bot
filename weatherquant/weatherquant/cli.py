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

from weatherquant.db.engine import get_engine
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

    import numpy as np

    from weatherquant.calibrate import evaluate, persist, strata

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
            # Group the aggregated pairs into (lead, month) strata.
            by_month: dict[int, list[strata.TrainingPair]] = {}
            for pair in pairs:
                by_month.setdefault(pair.month, []).append(pair)

            count = 0
            for month, month_pairs in sorted(by_month.items()):
                m = np.array([p.m for p in month_pairs], dtype=float)
                s2 = np.array([p.s2 for p in month_pairs], dtype=float)
                y = np.array([p.y for p in month_pairs], dtype=float)
                target_dates = [p.target_date for p in month_pairs]  # real verifying days (D-10)

                samples = strata.StratumSamples(
                    city=city, model=model, lead=lead, month=month, m=m, s2=s2, y=y
                )
                fit = strata.fit_stratum_pooled(samples)

                # OOS audit metrics on a genuine TEMPORAL split (D-10): order the stratum's real
                # target dates and hold out the latest oos_fraction. This requires >= 2 DISTINCT
                # dates — with fewer the split would be positional (look-ahead leakage), so we
                # leave the metrics NaN (absence = absence) rather than persist a leaky number.
                # The audit fit feeds the REAL ensemble variance s2 (sqrt(s2) is the raw-ensemble
                # baseline spread); the deterministic-collapse pseudo-member bug is gone.
                # The persisted params come from the FULL-stratum fit, so their data cutoff is the
                # latest target date — independent of the diagnostic OOS split.
                crps_train = crps_oos = crps_baseline_oos = math.nan
                trained_through = max(target_dates)  # persisted fit's data cutoff (D-13)
                if len(set(target_dates)) >= 2:
                    oos = evaluate.evaluate_stratum_oos_aggregated(
                        target_dates, m, s2, y, oos_fraction=oos_fraction
                    )
                    crps_train = oos.crps_train
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
    parser.error(f"unknown command: {args.command}")
    return 2  # unreachable — parser.error exits


if __name__ == "__main__":
    raise SystemExit(main())
