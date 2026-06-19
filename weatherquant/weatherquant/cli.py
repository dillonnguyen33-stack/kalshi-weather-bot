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
    # — closing the Phase-4 D-08/D-16 loop. It places NO real order (paper only,
    # assert_paper_mode) and persists the snapshot + (possibly partial) fill via the audited
    # market.persist path, stamped with the real WS event time.
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
        help="Use the fixed Kalshi DEMO hosts for the orderbook snapshot (SSRF-safe const).",
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


def _blend_distribution(
    bind: Any, city: str, target: date, lead: int
) -> dict[str, Any]:
    """Shared latest→predict→blend→sufficiency body (reused by run_price AND run_paper).

    Reads the latest calibration params + forecast members for ``(city, target, lead)``,
    aggregates members to ``(mean_f, var_f)`` at the ONE K→°F seam, reconstructs ``(μ_i, σ_i)``
    via ``link.predict``, computes accuracy weights from ``crps_oos`` and Vincentizes to
    ``(μ_blend, σ_blend)``, reads the AFD disagreement flag, and resolves the conservative
    (smallest-sufficiency-ramp) representative ``n_train``/``pool_level`` (WR-05). Fails loud on
    a NULL ``n_train`` (WR-02) and on no usable model. Pure money-path math stays in
    ``weatherquant.price``; this is the shared DB-read edge.

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
        # EV and the sized stake MUST share one decision basis (WR-01). bucket_ev shrinks the
        # model prob toward the market mid internally (D-08, p_used); size Kelly on that SAME
        # shrunk belief so the printed edge and the stake agree in sign near the boundary.
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
    (CR-01) — unit-consistent with ``best_yes_bid``/``best_no_bid`` (integer cents) and the
    fill's ``avg_price_cents``, so ``clv.clv_cents`` subtracts with NO conversion. The [0,1]
    pricing value is ``mid_cents / 100.0`` (see :func:`_reflection_midpoint`).

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
    return (best_yes_bid + best_yes_ask) / 2.0


def _reflection_midpoint(book: object) -> float:
    """Derive the live-book midpoint in [0,1] from the REFLECTED best levels (the ONE seam).

    The [0,1] PRICING value: ``_reflection_midpoint_cents(book) / 100.0``. This is the value
    the Phase-4 D-08/D-16 loop closes on (fed into ``price.p_used``/``bucket_ev``/
    ``stake_fraction``) — it MUST equal the reflection-derived mid, never a native ask read
    (there is none) and never a fabricated value. The PERSISTED ``mid`` is the un-divided
    cents value from :func:`_reflection_midpoint_cents` (CR-01); the two are the SAME midpoint
    in two units, split so persistence and pricing never share one mismatched unit.
    """
    return _reflection_midpoint_cents(book) / 100.0


def _snapshot_event_time(snapshot: Mapping[str, Any]) -> datetime:
    """Extract the snapshot's REAL WS event time, fail-loud (never now(), D-08).

    The persisted ``available_at`` MUST be the real observed instant of the book, never the
    wall clock (back-dating destroys Phase-6 no-look-ahead, D-08). Accepts an explicit
    ``event_time`` datetime or a ``snapshot_for`` ISO string; a naive value is treated as UTC.
    """
    value = snapshot.get("event_time")
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    raw = snapshot.get("snapshot_for")
    if isinstance(raw, str):
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    raise SystemExit(
        "paper: orderbook snapshot carries no WS event time (event_time/snapshot_for) — "
        "refusing to stamp a fill/snapshot with now() (D-08)."
    )


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
    7. PERSISTS the market snapshot (debounced cadence ``PAPER_SNAPSHOT_CADENCE_SECONDS``,
       dense enough that the CLV closing window holds >= 1 snapshot, PAP-04) and the fill via
       ``market.persist`` — both stamped with the REAL WS event time (D-08);
    8. places NO real order. Returns a small result dict (``midpoint``/``p_used``/``ev``/
       ``stake``/fill summary/persisted-snapshot event times).

    All pure math stays in :mod:`weatherquant.price` / :mod:`weatherquant.market.fills`; this
    is the I/O edge.
    """
    import asyncio

    from weatherquant import price as pricing
    from weatherquant.market import fills, reflect

    # The debounced snapshot cadence MUST stay strictly finer than the CLV closing window so the
    # window always holds >= 1 persisted snapshot (PAP-04 cadence sufficiency, T-05-20).
    assert PAPER_SNAPSHOT_CADENCE_SECONDS < clv.CLV_WINDOW_MINUTES * 60, (
        "PAPER_SNAPSHOT_CADENCE_SECONDS must be strictly finer than the CLV closing window "
        "(clv.CLV_WINDOW_MINUTES) so the window is never silently sparse (PAP-04)."
    )

    city = args.city
    target = args.date
    lead = args.lead
    ticker = args.ticker

    settings = get_settings()
    # Paper-only: the simulator is unreachable in validated 'live' mode (the live path would use
    # the real order flow, not the shadow simulator). assert_paper_mode is the fail-CLOSED fence
    # for the (non-existent) order path; here we assert we are NOT in live so the simulator runs.
    if settings.execution_mode == "live":
        raise SystemExit(
            "paper: execution_mode='live' — the paper-fill simulator does not run in live "
            "mode (no order-submission path exists this milestone, D-15/T-05-14)."
        )

    bind = get_engine()
    signer = KalshiSigner.from_settings(settings)

    import httpx

    async def _fetch() -> dict[str, Any]:
        async with httpx.AsyncClient() as http:
            rest_host = _demo_rest_host() if getattr(args, "demo", False) else None
            if rest_host is not None:
                return await fetch_snapshot(http, signer.sign, ticker, rest_host=rest_host)
            return await fetch_snapshot(http, signer.sign, ticker)

    snapshot = asyncio.run(_fetch())
    event_time = _snapshot_event_time(snapshot)

    # The REAL reflection-derived live-book midpoint, derived ONCE and split into TWO units so
    # persistence and pricing never share one mismatched unit (CR-01): mid_cents is the
    # float-valued CENTS midpoint PERSISTED as market_snapshots.mid (unit-consistent with
    # best_*_bid/avg_price_cents → CLV needs no conversion); mid_unit = mid_cents/100.0 is the
    # [0,1] PRICING value the Phase-4 D-08/D-16 loop closes on (p_used/EV/Kelly). The pricing
    # value is UNCHANGED from before — only the persisted unit is corrected.
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

    # Persist the snapshot at the debounced cadence (here: once per run_paper invocation, which
    # is by construction within one cadence window), stamped with the REAL WS event time (D-08)
    # and carrying the load-bearing book fields + raw book JSONB (D-03). Persisting on every
    # invocation keeps the CLV closing window dense (PAP-04): a book change inside the window
    # lands a snapshot whose available_at is inside the window.
    best_yes_bid = max((int(p) for p, _ in snapshot.get("yes") or []), default=None)
    best_no_bid = max((int(p) for p, _ in snapshot.get("no") or []), default=None)
    # The per-snapshot volume signal is the total RESTING TOP-OF-BOOK liquidity present at this
    # observed instant: the summed resting size across both bid sides of the orderbook payload
    # (each level is [price_cents, size]). This is a REAL value off the feed — the orderbook
    # payload exposes resting SIZE, not a separate traded-volume field — so the CLV closing mid
    # is genuinely weighted by the book liquidity present at each snapshot (D-09, WR-01), not a
    # fabricated placeholder. Cast to int (whole contracts).
    volume = int(
        sum(int(sz) for _, sz in (snapshot.get("yes") or []))
        + sum(int(sz) for _, sz in (snapshot.get("no") or []))
    )
    snapshot_for = event_time.isoformat()
    persisted_snapshot_times: list[str] = []
    rc_snap = persist_snapshot(
        bind,
        ticker=ticker,
        snapshot_for=snapshot_for,
        best_yes_bid=best_yes_bid,
        best_no_bid=best_no_bid,
        # PERSIST CENTS (CR-01): mid_cents is unit-consistent with best_*_bid/avg_price_cents so
        # CLV subtracts directly. NEVER persist the [0,1] mid_unit here.
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
            persist_fill(
                bind,
                ticker=ticker,
                trade_id=trade_id,
                side="yes",
                price=int(round(fill.avg_price_cents)),
                count=fill.count,
                fee=int(round(pricing.exact_fee(fill.count, mid_unit) * 100)),
                is_maker=fill.is_maker,
                event_time=fill.event_time,
                bucket_prob=prob,
                ev=ev,
                kelly_stake=stake,
                detail={"avg_price_cents": fill.avg_price_cents, "partial": fill.partial},
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
        # The [0,1] pricing midpoint (UNCHANGED loop-closure value — the spy asserts this == the
        # value fed into p_used); mid_cents is the persisted-unit twin (CR-01).
        "midpoint": mid_unit,
        "mid_cents": mid_cents,
        "p_used": pu,
        "prob": prob,
        "ev": ev,
        "stake": stake,
        "fill": fill_summary,
        "persisted_snapshot_times": persisted_snapshot_times,
    }


def _demo_rest_host() -> str:
    """Return the fixed Kalshi DEMO REST host constant (SSRF guard; never untrusted input)."""
    from weatherquant.market.client import REST_HOST_DEMO

    return REST_HOST_DEMO


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
