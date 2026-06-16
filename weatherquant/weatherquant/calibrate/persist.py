"""Append-only persistence for fitted calibration params + audit metrics (D-13 / SYS-01).

A refit is a FRESH INSERT — never a mutate-in-place. This module mirrors
``ingest/writer.py``'s single audited write contract for the ``calibration_params`` table:
split the natural key ``(city, model, lead, month)`` from the payload content, execute a
SQLAlchemy **Core** ``calibration_params.insert().values(...)``, and raise on
``rowcount != 1`` via an EXPLICIT raise (not a bare ``assert`` — the guard must survive
``python -O`` / PYTHONOPTIMIZE, WR-06). It REUSES ``writer.WriteIntegrityError`` and the
``writer.Bind`` alias rather than re-declaring calibrate-local equivalents, so there is exactly
one integrity-error type and one bind contract across every write path.

**Append-only (D-13, Phase-1 D-10).** There is NO in-place-mutation / merge / conflict-clause
path here: the append-only trigger on ``calibration_params`` would raise, so a correction/refit
INSERTs a new row with a later ``available_at`` and ``latest()`` returns the current params. The
source-scan guard for those forbidden write verbs over this file is therefore 0 (INSERT only).

**Point-in-time (D-13).** ``available_at`` is a function PARAMETER — the training-run completion
instant supplied by the caller — never computed via ``now()`` inside this function (that would
back-date knowledge to the wrong moment and break the no-look-ahead invariant Phase 6 depends
on). ``trained_through`` is the DATA cutoff (the last train ``target_date``) so Phase 6 can
re-derive any historical fit (D-13).

**Injection-safe (T-03-05 / ASVS V5).** Columns are resolved via ``calibration_params.c[...]``
(SQLAlchemy Core); no caller string is ever f-string-interpolated into SQL. Values are bound
parameters. The CLI validates city/model/date BEFORE any call reaches here.

Pure SQLAlchemy Core — no scipy/sklearn (the AST guard fences the calibration package).
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

    Mirrors ``ingest/writer.py::_insert_row``: the natural key ``(city, model, lead, month)``
    plus the interpretable EMOS/NGR payload (``mean_intercept``/``mean_slope`` = (a, b);
    ``var_intercept``/``var_slope`` = (c, d); ``sigma_floor``; ``n_train``; ``pool_level``) and
    the audit metrics (``crps_train``/``crps_oos``/``crps_baseline_oos``) are written via a Core
    ``calibration_params.insert().values(...)``. The single-row integrity guard is an EXPLICIT
    raise of :class:`weatherquant.ingest.writer.WriteIntegrityError` on ``rowcount != 1`` — NOT a
    bare ``assert`` (so it survives ``python -O``, WR-06).

    A refit is a fresh INSERT (append-only — the trigger raises on any mutate-in-place, D-13).
    There is no skip-before-insert here (unlike the forecast/observation writer): each fit is a
    new point-in-time record, distinguished by its later ``available_at``.

    Args:
        bind: a SQLAlchemy ``Engine`` or ``Connection`` (the ``writer.Bind`` alias). Built by
            :func:`weatherquant.db.engine.get_engine` so ``preserve_rowcount`` holds (a single-row
            insert reports ``rowcount == 1`` despite the implicit ``RETURNING id``, D-11).
        city, model, lead, month: the natural key (``month`` = calendar month of the strata).
        mean_intercept, mean_slope: EMOS mean params (a, b: μ = a + b·m).
        var_intercept, var_slope: EMOS variance params (c, d: σ² = max(σ_floor², c² + d²·s²)).
        sigma_floor: the °F σ-floor (D-09).
        n_train: number of training samples used for the fit.
        pool_level: pooling-ladder provenance (``"month"`` / ``"shrunk:<rung>"`` / ``"parent:<rung>"``).
        crps_train, crps_oos, crps_baseline_oos: audit metrics (in-sample / OOS calibrated /
            OOS raw-ensemble baseline, D-11).
        trained_through: the DATA cutoff — the last train ``target_date`` (D-13).
        available_at: the training-run completion instant (point-in-time of knowledge, D-13);
            supplied by the caller, NEVER ``now()``-ed inside this function.

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
        # Explicit raise (WR-06): `python -O` strips bare asserts, which would silently disable
        # the only check that the single audited INSERT actually landed a row. preserve_rowcount
        # (engine.get_engine) makes a single-row insert report 1 despite the RETURNING id (D-11).
        if result.rowcount != 1:
            raise WriteIntegrityError(
                f"expected rowcount==1 inserting into calibration_params, "
                f"got {result.rowcount}"
            )
        return int(result.rowcount)

    # Match the writer's bind handling: an Engine opens its own transaction; a Connection is used
    # directly (the caller owns the transaction). Imported here to keep the module import-light.
    from sqlalchemy.engine import Engine

    if isinstance(bind, Engine):
        with bind.begin() as conn:
            return _do(conn)
    return _do(bind)
