"""``weatherquant price`` — latest→predict→blend→bucket/EV/Kelly smoke command (D-15/D-16).

``get_engine``/``get_settings`` are imported into this module's namespace so the run + blend
bodies resolve ``cli.pricing.get_engine`` / ``cli.pricing.get_settings`` (the seams tests
monkeypatch). ``_blend_distribution`` is the shared latest→blend→sufficiency body reused by
``run_paper`` — it lives here and ``cli.paper`` imports it.
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from typing import Any

from weatherquant.db.engine import get_engine, get_settings

logger = logging.getLogger(__name__)


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


def _price_bucket(
    blend: dict[str, Any], ticker: str, mid: float
) -> tuple[float, float, float, float]:
    """Price one bucket on ``blend`` at ``mid`` — the shared run_price/run_paper money tail.

    Returns ``(prob, p_used, ev, stake_fraction)``. The ONLY difference between the mocked
    (``--market-mid``) and real (live ``mid_unit``) money paths is the ``mid`` arg, so both
    share this one stake-arg ordering — EV and the sized stake stay on the SAME shrunk belief
    (``p_used``) and can never silently disagree in sign near the boundary (D-08/D-16).
    """
    from weatherquant import price as pricing

    lo, hi, open_lo, open_hi = pricing.parse_ticker(ticker)
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
        # The shared money tail (also run by run_paper); the ONLY difference there is the REAL
        # live-book midpoint replacing this MOCKED --market-mid (D-08/D-16). EV + stake share the
        # one p_used basis inside _price_bucket so the printed edge and the stake never disagree.
        prob, _pu, ev, stake = _price_bucket(blend, args.ticker, market_mid)
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
