"""``weatherquant price`` — latest→predict→blend→bucket/EV/Kelly smoke command (D-15/D-16).

``get_engine``/``get_settings`` are imported into this module's namespace so the run + blend
bodies resolve ``cli.pricing.get_engine`` / ``cli.pricing.get_settings`` (the seams tests
monkeypatch). ``_blend_distribution`` is the shared latest→blend→sufficiency body reused by
``run_paper`` — it lives here and ``cli.paper`` imports it.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections.abc import Mapping
from datetime import date
from typing import Any

from weatherquant.db.engine import get_engine, get_settings
# Imported INTO this module's namespace as test seams (run_price's structured-strike fetch builds
# the signer + resolves the host here, so tests monkeypatch cli.pricing.KalshiSigner /
# cli.pricing._resolve_hosts). cli/ sits at the edge — importing market here breaks no invariant
# (the no-market-import rule is price/-only); cli.paper already imports the same seams.
from weatherquant.market.auth import KalshiSigner
from weatherquant.market.client import _resolve_hosts

logger = logging.getLogger(__name__)

# Plausible Fahrenheit daily-high strike band + integer tolerance for the structured-strike
# scale-sanity check. The 05.1-UAT demo returns SCALED strikes (e.g. 8.5e-05 for 85°F); int()-ing
# that silently yields 0 and prices the WRONG bucket. A real KXHIGH strike is a whole °F degree in
# this generous band (deep-winter Denver/Chicago lows to Phoenix/Austin highs), so a non-integer or
# out-of-band value FAILS LOUD rather than int()-zeroing a degenerate strike (D-06 fail-loud).
_STRIKE_MIN_F = -80
_STRIKE_MAX_F = 140
_STRIKE_INT_TOL = 1e-6


def _plausible_strike(value: Any, *, field: str, ticker: str) -> int | None:
    """Validate a structured strike is a plausible integer °F degree, fail loud on scaled junk.

    ``None`` passes through (an open-tail strike legitimately omits one of floor/cap). A present
    strike must be a finite, (near-)integer value inside the plausible daily-high band; the demo's
    ``8.5e-05`` is non-integer (``8.5e-05`` is not within ``_STRIKE_INT_TOL`` of ``0``) so it raises
    instead of silently becoming ``int(8.5e-05) == 0`` — the exact degenerate-strike money-path
    hazard the project's fail-loud discipline forbids (never int()-zero).
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(
            f"price/paper: {field}={value!r} on {ticker} is not numeric — refusing to price a "
            "non-numeric strike (fail loud, never int()-zero)."
        ) from exc
    if not math.isfinite(f):
        raise SystemExit(
            f"price/paper: {field}={value!r} on {ticker} is not a finite strike (fail loud)."
        )
    rounded = round(f)
    if abs(f - rounded) > _STRIKE_INT_TOL:
        raise SystemExit(
            f"price/paper: {field}={value!r} on {ticker} is not an integer °F strike (looks "
            "scaled, e.g. the demo's 8.5e-05 for 85°) — refusing to int()-zero a degenerate "
            "strike (D-06 fail-loud)."
        )
    if not (_STRIKE_MIN_F <= rounded <= _STRIKE_MAX_F):
        raise SystemExit(
            f"price/paper: {field}={value!r} on {ticker} rounds to {rounded}°F, outside the "
            f"plausible daily-high band [{_STRIKE_MIN_F}, {_STRIKE_MAX_F}] — fail loud (D-06)."
        )
    return int(rounded)


def _resolve_bucket(
    ticker: str, *, market: Mapping[str, Any] | None = None
) -> tuple[int | None, int | None, bool, bool]:
    """Resolve ``(lo, hi, open_lo, open_hi)`` ONCE, preferring the structured GetMarket strikes (D-06).

    The structured ``floor_strike`` / ``cap_strike`` / ``strike_type`` from a fetched ``GetMarket``
    record is the AUTHORITATIVE, live-confirmed path (05-UAT Test 3 / test_kxhigh_live_crosscheck).
    REAL date-coded KXHIGH tickers do NOT match the positional ``KXHIGH{SUFFIX}-lo-hi`` regex, so
    they REQUIRE the structured record — the ``B``/``T`` tail direction is unconfirmed and MUST NOT
    be guessed on the money path. When no record is supplied the only safe fallback is the
    positional ticker-string parse (the synthetic closed-range form); a real ticker with no record
    fails loud inside ``parse_ticker`` rather than fabricating a bucket.
    """
    from weatherquant import price as pricing

    if market is not None:
        floor_strike = _plausible_strike(
            market.get("floor_strike"), field="floor_strike", ticker=ticker
        )
        cap_strike = _plausible_strike(
            market.get("cap_strike"), field="cap_strike", ticker=ticker
        )
        return pricing.parse_ticker(
            floor_strike=floor_strike,
            cap_strike=cap_strike,
            strike_type=market.get("strike_type"),
        )
    return pricing.parse_ticker(ticker)


def _needs_market_record(ticker: str) -> bool:
    """True iff ``ticker`` is NOT the positional synthetic form (so a structured record is needed).

    A positional ``KXHIGH{SUFFIX}-lo-hi`` ticker resolves with NO network (the synthetic
    closed-range form tests/dev use). A REAL date-coded ticker raises ``ValueError`` in
    ``parse_ticker`` — that is precisely the branch that must fetch the structured ``GetMarket``
    record (never guess the B/T grammar).
    """
    from weatherquant import price as pricing

    try:
        pricing.parse_ticker(ticker)
    except ValueError:
        return True
    return False


def _fetch_market_record(
    ticker: str, sign: Any, rest_host: str
) -> dict[str, Any]:
    """Run the signed async ``GetMarket`` fetch synchronously (the I/O edge; structured-strike path).

    Shared by ``run_price`` and ``run_paper`` so both resolve the structured strikes through the
    ONE :func:`weatherquant.market.client.fetch_market` seam (no re-derived auth/host).
    """
    import asyncio

    import httpx

    from weatherquant.market.client import fetch_market

    async def _go() -> dict[str, Any]:
        async with httpx.AsyncClient() as http:
            return await fetch_market(http, sign, ticker, rest_host=rest_host)

    return asyncio.run(_go())


def _resolve_bucket_for_run(
    ticker: str, *, sign: Any, rest_host: str
) -> tuple[int | None, int | None, bool, bool]:
    """Resolve bucket edges ONCE for a run: structured GetMarket strikes for a real ticker, else positional.

    The structured fetch is reached ONLY for a real date-coded ticker (the positional synthetic
    form resolves offline with no signer/network) so the offline DB-free suite stays green while a
    live KXHIGH ticker drives the authoritative structured path.
    """
    if not _needs_market_record(ticker):
        return _resolve_bucket(ticker)
    market = _fetch_market_record(ticker, sign, rest_host)
    return _resolve_bucket(ticker, market=market)


def _resolve_price_bucket(
    ticker: str, *, settings: Any, demo: bool
) -> tuple[int | None, int | None, bool, bool]:
    """``run_price`` bucket resolution: structured GetMarket strikes for a real ticker, else positional.

    Builds the signer + resolves the host ONLY when a REAL date-coded ticker forces the structured
    fetch (the synthetic positional form resolves offline with no signer/network), so the smoke
    command stays offline for the tests/dev synthetic ticker and reaches the network only for a live
    KXHIGH market. The mid stays MOCKED (``--market-mid``, D-16); only the bucket edges come from the
    live record.
    """
    if not _needs_market_record(ticker):
        return _resolve_bucket(ticker)
    _, rest_host = _resolve_hosts(demo)
    signer = KalshiSigner.from_settings(settings)
    return _resolve_bucket_for_run(ticker, sign=signer.sign, rest_host=rest_host)


_FIT_PARAM_KEYS = ("mean_intercept", "mean_slope", "var_intercept", "var_slope", "sigma_floor")


def _has_usable_fit(row: Any) -> bool:
    """True when a calibration row carries a complete (all-non-NULL) EMOS param set.

    A degenerate placeholder row (every param NULL) is unusable — ``link.predict`` can't run on
    it. The nearest-month fallback must ignore such rows so a junk month never shadows a real fit;
    the pricing loop applies the SAME check per model. One definition, two call sites.
    """
    return all(row[k] is not None for k in _FIT_PARAM_KEYS)


def _nearest_month(target_month: int, available: list[int]) -> int:
    """Pick the month in ``available`` closest to ``target_month`` on the cyclic 1..12 axis.

    Circular distance ``min(d, 12 - d)`` for ``d = abs(m - target_month)`` so Dec(12)↔Jan(1) == 1.
    Deterministic tie-break: on equal distance, prefer the LOWER month number (the ``m`` in the key
    tuple). Caller guarantees ``available`` is non-empty. Pure int arithmetic — no dependency added.
    """
    return min(
        available,
        key=lambda m: (min(abs(m - target_month), 12 - abs(m - target_month)), m),
    )


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
    from weatherquant.ingest.afd import SOURCE as AFD_SOURCE

    cap = get_settings().max_position_fraction
    month = target.month
    forecasts = queries.latest(
        bind, "forecasts", where={"city": city, "target_date": target, "lead": lead}
    )
    used_month = month
    cal_rows = queries.latest(
        bind, "calibration_params", where={"city": city, "lead": lead, "month": month}
    )
    if not cal_rows:
        # The exact month has no fit. Broaden to (city, lead) — one row per (model, month) — and
        # price with the NEAREST fitted month on the cyclic 1..12 axis. Only months with a USABLE
        # (all-params-present) fit are candidates, so a degenerate NULL-param placeholder can't
        # shadow a real fit a month further away. A missing (city, lead) — or one with only junk
        # rows — leaves cal_rows empty so the downstream fail-loud guard still fires: the fallback
        # rescues a missing MONTH only, never a missing city/lead.
        all_rows = [
            r
            for r in queries.latest(bind, "calibration_params", where={"city": city, "lead": lead})
            if _has_usable_fit(r)
        ]
        available = {int(r["month"]) for r in all_rows}
        if available:
            used_month = _nearest_month(month, sorted(available))
            cal_rows = [r for r in all_rows if int(r["month"]) == used_month]
            logger.warning(
                "calibration month=%s absent for city=%s lead=%s — pricing with nearest "
                "fitted month=%s (%d fits)",
                month, city, lead, used_month, len(cal_rows),
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
        if cal is None or not vals or not _has_usable_fit(cal):
            continue
        arr = np.asarray(vals, dtype=np.float64)
        mean_f = float(arr.mean())
        var_f = float(arr.var()) if arr.size > 1 else 0.0
        params = tuple(cal[k] for k in _FIT_PARAM_KEYS)
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
            f"city={city} date={target} lead={lead} month={used_month}."
        )

    weights = pricing.accuracy_weights(np.asarray(crps_oos, dtype=np.float64))
    mu_blend, sigma_blend = pricing.blend_gaussians(
        np.asarray(mus), np.asarray(sigmas), weights
    )

    afd_rows = queries.latest(
        bind, "observations", where={"city": city, "target_date": target, "source": AFD_SOURCE}
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


def _price_bucket(
    blend: dict[str, Any],
    bucket: tuple[int | None, int | None, bool, bool],
    mid: float,
) -> tuple[float, float, float, float]:
    """Price one bucket on ``blend`` at ``mid`` — the shared run_price/run_paper money tail.

    Takes the bucket edges RESOLVED ONCE (``_resolve_bucket`` — structured GetMarket strikes for a
    real date-coded ticker, the positional parse for the synthetic form) rather than re-parsing the
    ticker string here, so a real KXHIGH ticker the positional regex cannot match is priced via the
    authoritative structured path. Returns ``(prob, p_used, ev, stake_fraction)``. The ONLY
    difference between the mocked (``--market-mid``) and real (live ``mid_unit``) money paths is the
    ``mid`` arg, so both share this one stake-arg ordering — EV and the sized stake stay on the SAME
    shrunk belief (``p_used``) and can never silently disagree in sign near the boundary (D-08/D-16).
    """
    from weatherquant import price as pricing

    lo, hi, open_lo, open_hi = bucket
    c_lo, c_hi = pricing.integers_in_bucket(lo, hi, open_lo, open_hi)
    prob = pricing.bucket_prob(
        blend["mu_blend"], blend["sigma_blend"], c_lo, c_hi, open_lo, open_hi
    )
    pu = pricing.p_used(prob, mid)
    ev = pricing.bucket_ev(prob, mid, mid)
    stake = pricing.stake_fraction(
        pu, mid, pricing.exact_fee(1, mid),
        blend["sigma_blend"], blend["n_train"], blend["pool_level"], blend["afd_flag"],
        cap=blend["cap"],
    )
    return prob, pu, ev, stake


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
        # Resolve the bucket edges ONCE: structured GetMarket strikes for a real date-coded ticker
        # the positional regex cannot match, else the synthetic positional parse (no network). The
        # mid stays MOCKED (--market-mid, D-16); only the bucket edges come from the live record.
        bucket_edges = _resolve_price_bucket(
            args.ticker, settings=get_settings(), demo=bool(getattr(args, "demo", False))
        )
        # The shared money tail (also run by run_paper); the ONLY difference there is the REAL
        # live-book midpoint replacing this MOCKED --market-mid (D-08/D-16). EV + stake share the
        # one p_used basis inside _price_bucket so the printed edge and the stake never disagree.
        prob, _pu, ev, stake = _price_bucket(blend, bucket_edges, market_mid)
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
