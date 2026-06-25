"""Post-Gate-1 calibration drift monitor (SYS-02 / D-12): trailing-window ECE breach → non-zero exit.

D-12 (verify subtree-local; drift monitor — log + non-zero exit, no UI): once Gate-1 has passed,
``monitor_drift`` watches the trailing-window calibration. It replays the ledger as-of-correctly via
the existing :func:`verify.backtest.walk_forward` (no new temporal/calibration math — D-04) over the
trailing ``window_days`` and scores the per-(city, model) reliability error by REUSING
:func:`verify.metrics.ece_equal_count` (the equal-count ECE scalar — never a re-derived metric). If
any stratum's trailing ECE exceeds ``Settings.drift_reliability_threshold`` (read from config, NEVER
hardcoded), the monitor logs at ERROR (naming the city/model/value/threshold) and returns a NON-ZERO
process exit code so cron/systemd treats a calibration breach as a real failure (SYS-02). Under
threshold for every stratum it returns 0.

No Discord/web/email alerting (out of scope — D-12); the surface is exactly the ERROR log + the
non-zero exit. Reads the ledger via the injected ``bind``; the threshold via the injected
``settings``. ``tests/test_drift.py`` pins the contract (the Settings-knob unit test is DB-free; the
ledger-backed breach/clear cases are ``integration``-marked).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, UTC

import numpy as np

# The as-of-correct replay is REUSED verbatim (no new temporal/calibration math — D-04). Imported
# at module top as the monitor's test seam (``tests/test_drift.py`` patches ``drift.walk_forward``
# to feed synthetic paired records); backtest pulls in only numpy + sqlalchemy + project modules,
# so drift stays outside the scipy/sklearn/matplotlib AST-guarded metric core.
from weatherquant.verify.backtest import walk_forward

# The reliability error REUSES the one equal-count ECE scalar — never a re-derived metric (D-04).
from weatherquant.verify.metrics import ece_equal_count

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_DRIFT_WINDOW_DAYS", "monitor_drift"]

#: Default trailing window for the drift monitor when the caller passes none — 30 calendar days is
#: a sensible cron cadence (roughly a month of settled outcomes) without over-smoothing a recent
#: calibration breach. Operator-overridable via ``--window-days``.
DEFAULT_DRIFT_WINDOW_DAYS: int = 30


def monitor_drift(bind, settings, *, window_days=None, cities, models) -> int:
    """Trailing-window calibration-drift monitor → process exit code (SYS-02 / D-12).

    For each ``(city, model)`` in the requested strata this replays the trailing ``window_days``
    of the ledger via :func:`verify.backtest.walk_forward` (as-of-correct, no new math — D-04),
    pools the resulting paired records' ``(wq_prob, o_i)`` and scores the reliability error with
    :func:`verify.metrics.ece_equal_count` (the one equal-count ECE scalar — never re-derived). It
    compares each stratum's trailing ECE to ``settings.drift_reliability_threshold`` (read from
    Settings, NEVER hardcoded). On ANY breach it logs at ERROR (naming the offending
    city/model/value/threshold) and returns a NON-ZERO exit code so cron/systemd surfaces it
    (SYS-02); when every stratum is under threshold (or has no settled data to score) it returns 0.
    """
    # The threshold ALWAYS comes from Settings — a hardcoded fallback would defeat the operator
    # policy knob (D-12). A missing attribute is a configuration bug, surfaced loud.
    threshold = float(settings.drift_reliability_threshold)

    window = int(window_days) if window_days is not None else DEFAULT_DRIFT_WINDOW_DAYS
    if window <= 0:
        raise ValueError(f"window_days must be strictly positive, got {window!r}")

    end_day: date = datetime.now(UTC).date()
    start_day: date = end_day - timedelta(days=window)

    breached = False
    for city in cities:
        for model in models:
            records, _coverage = walk_forward(
                bind, city, model, lead=0, start=start_day, end=end_day, oos_slice=None
            )
            scored = [r for r in records if r.excluded_reason is None]
            if not scored:
                # No settled outcomes in the trailing window for this stratum — nothing to score,
                # so nothing can breach (absence is absence; no fabricated reliability number).
                logger.info(
                    "drift: no settled records for city=%s model=%s in the trailing %d-day "
                    "window — skipping",
                    city,
                    model,
                    window,
                )
                continue

            f = np.array([r.wq_prob for r in scored], dtype=float)
            o = np.array([r.o_i for r in scored], dtype=float)
            ece = ece_equal_count(f, o)

            if ece > threshold:
                breached = True
                logger.error(
                    "DRIFT BREACH: city=%s model=%s trailing-%dd reliability error %.4f exceeds "
                    "threshold %.4f (Settings.drift_reliability_threshold) — calibration has "
                    "degraded (SYS-02).",
                    city,
                    model,
                    window,
                    ece,
                    threshold,
                )
            else:
                logger.info(
                    "drift OK: city=%s model=%s trailing-%dd reliability error %.4f <= %.4f",
                    city,
                    model,
                    window,
                    ece,
                    threshold,
                )

    # Non-zero exit on ANY breach so a scheduler/operator treats it as a real failure (SYS-02).
    return 1 if breached else 0
