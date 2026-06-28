"""``weatherquant calibrate`` — fit + persist EMOS/NGR params per stratum (D-13 / CAL-01/03).

``get_engine`` is imported into this module's namespace so the run body resolves
``cli.calibrate.get_engine``; the heavy ``weatherquant.calibrate`` imports stay lazy in the body.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

from weatherquant.db.engine import get_engine

from ._args import _resolve_cities, _resolve_models

logger = logging.getLogger(__name__)


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
    from weatherquant.calibrate import crps, evaluate, link, persist, strata

    models = _resolve_models(args)
    cities = _resolve_cities(args)
    lead = args.lead
    oos_fraction = args.oos_fraction

    bind = get_engine()
    available_at = datetime.now(UTC)  # training-run completion instant (D-13)
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
                crps_oos = crps_baseline_oos = float("nan")
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
