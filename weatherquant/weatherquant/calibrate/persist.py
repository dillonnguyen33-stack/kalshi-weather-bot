"""Append-only persistence for fitted calibration params + audit metrics (D-13 / SYS-01).

A refit is a FRESH INSERT — never a mutate-in-place (the append-only trigger raises otherwise);
``latest()`` returns the row with the newest ``available_at``. Mirrors ``ingest/writer.py``'s
audited write contract and reuses its ``WriteIntegrityError``/``Bind`` so there is one
integrity-error type across every write path.

* Point-in-time (D-13): ``available_at`` is a PARAMETER (the training-run completion instant),
  never ``now()``-ed inside — that would back-date knowledge and break no-look-ahead.
  ``trained_through`` is the data cutoff so Phase 6 can re-derive any historical fit.
* Injection-safe (T-03-05): columns resolve via ``calibration_params.c[...]``; no caller string
  is f-stringed into SQL.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from weatherquant.db.models import calibration_params
from weatherquant.ingest.writer import Bind, WriteIntegrityError

__all__ = ["store_calibration_params"]

logger = logging.getLogger(__name__)


def store_calibration_params(
    bind: Bind,
    *,
    city: str,
    model: str,
    lead: int,
    month: int,
    mean_intercept: float,
    mean_slope: float,
    var_intercept: float,
    var_slope: float,
    sigma_floor: float,
    n_train: int,
    pool_level: str,
    crps_train: float,
    crps_oos: float,
    crps_baseline_oos: float,
    trained_through: date,
    available_at: datetime,
) -> int:
    """INSERT one fitted ``calibration_params`` row through the append-only contract (D-13).

    Mirrors ``ingest/writer.py::_insert_row``: the natural key plus the EMOS/NGR payload and audit
    metrics are written via a Core ``insert().values(...)``, with an EXPLICIT
    :class:`weatherquant.ingest.writer.WriteIntegrityError` raise on ``rowcount != 1`` (not a bare
    ``assert``, so it survives ``python -O``, WR-06). Each fit is a new point-in-time record
    distinguished by its later ``available_at`` (append-only — no skip-before-insert).

    Args:
        bind: a SQLAlchemy ``Engine`` or ``Connection`` (``writer.Bind``). Built by
            :func:`weatherquant.db.engine.get_engine` so ``preserve_rowcount`` holds (D-11).
        city, model, lead, month: the natural key (``month`` = calendar month of the strata).
        mean_intercept, mean_slope: EMOS mean params (a, b: μ = a + b·m).
        var_intercept, var_slope: EMOS variance params (c, d: σ² = max(σ_floor², c² + d²·s²)).
        sigma_floor: the °F σ-floor (D-09).
        n_train: number of training samples used for the fit.
        pool_level: pooling-ladder provenance (``"month"`` / ``"shrunk:<rung>"`` / ``"parent:<rung>"``).
        crps_train, crps_oos, crps_baseline_oos: audit metrics (in-sample / OOS calibrated / OOS
            raw-ensemble baseline, D-11). See the WR-05 caveat on
            :class:`weatherquant.calibrate.evaluate.OOSResult`: the held-out pair describes a
            different (UNPOOLED) fit than the persisted-fit ``crps_train``.
        trained_through: the DATA cutoff — the last train ``target_date`` (D-13).
        available_at: the training-run completion instant (D-13); caller-supplied, NEVER
            ``now()``-ed inside this function.

    Returns:
        ``1`` — exactly one row was inserted (the guard raises otherwise).
    """
    natural_key = {"city": city, "model": model, "lead": lead, "month": month}
    content = {
        "mean_intercept": mean_intercept,
        "mean_slope": mean_slope,
        "var_intercept": var_intercept,
        "var_slope": var_slope,
        "sigma_floor": sigma_floor,
        "n_train": n_train,
        "pool_level": pool_level,
        "crps_train": crps_train,
        "crps_oos": crps_oos,
        "crps_baseline_oos": crps_baseline_oos,
        "trained_through": trained_through,
    }
    values = {**natural_key, **content, "available_at": available_at}

    def _do(conn: object) -> int:
        result = conn.execute(calibration_params.insert().values(**values))  # type: ignore[attr-defined]
        # Explicit raise (WR-06): `python -O` strips bare asserts. preserve_rowcount
        # (engine.get_engine) makes a single-row insert report 1 despite the RETURNING id (D-11).
        if result.rowcount != 1:
            raise WriteIntegrityError(
                f"expected rowcount==1 inserting into calibration_params, "
                f"got {result.rowcount}"
            )
        return int(result.rowcount)

    # Match the writer's bind handling: an Engine opens its own transaction; a Connection is used
    # directly (the caller owns it). Imported here to keep the module import-light.
    from sqlalchemy.engine import Engine

    if isinstance(bind, Engine):
        with bind.begin() as conn:
            return _do(conn)
    return _do(bind)
