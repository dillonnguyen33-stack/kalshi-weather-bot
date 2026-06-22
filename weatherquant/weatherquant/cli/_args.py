"""Argparse surface for the ``weatherquant`` CLI — parser build + arg resolvers (stdlib only).

The parser, the ``type=`` validators, the model/city selector blocks, and the ``_resolve_*``
helpers live here so each subcommand module imports only the resolvers it needs. Behaviour and
``--help`` text are byte-identical to the pre-split single module — the only consolidation is the
``_add_model_selector``/``_add_city_selector`` helpers folding the ingest+calibrate duplication.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime

from weatherquant.ingest import orchestrator
from weatherquant.registry import CITIES, get_city

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


def _city_type(value: str) -> str:
    """argparse ``type=`` validating a city via :func:`get_city`; unknown → arg error (ASVS V5 / T-02-17)."""
    try:
        get_city(value)  # raises KeyError on an unknown code.
    except KeyError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def _add_model_selector(p: argparse.ArgumentParser, *, verb: str) -> None:
    """Add the required model/--all-models mutually-exclusive block (ingest+calibrate share it).

    ``verb`` is the ingest/calibrate word in the help text, the ONLY copy that differs between
    the two subparsers — collapsing the otherwise byte-identical duplication.
    """
    model_group = p.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        choices=ALL_MODELS,
        help=f"A single model/source to {verb} (e.g. hrrr, gfs, nws, openmeteo).",
    )
    model_group.add_argument(
        "--all-models",
        action="store_true",
        help=f"{verb.capitalize()} every GRIB model and supplementary source.",
    )


def _add_city_selector(p: argparse.ArgumentParser, *, verb: str) -> None:
    """Add the required city/--all-cities mutually-exclusive block (ingest+calibrate share it).

    ``verb`` is the ingest/calibrate word in the ``--all-cities`` help text, the ONLY copy that
    differs between the two subparsers.
    """
    city_group = p.add_mutually_exclusive_group(required=True)
    city_group.add_argument(
        "--city",
        type=_city_type,
        help=f"A single Kalshi city code (one of: {', '.join(sorted(CITIES))}).",
    )
    city_group.add_argument(
        "--all-cities",
        action="store_true",
        help=f"{verb.capitalize()} every registry city.",
    )


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

    _add_model_selector(ingest, verb="ingest")
    _add_city_selector(ingest, verb="ingest")

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

    _add_model_selector(calibrate, verb="calibrate")
    _add_city_selector(calibrate, verb="calibrate")

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
