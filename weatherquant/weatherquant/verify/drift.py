"""Post-Gate-1 calibration drift monitor (SYS-02 / D-10): trailing-window ECE breach → non-zero exit.

D-10 (verify subtree-local): once Gate-1 has passed, ``monitor_drift`` watches the trailing-window
calibration. If the per-(city, model) trailing ECE exceeds ``Settings.drift_reliability_threshold``
(read from config, NEVER hardcoded), the monitor logs at ERROR and returns a NON-ZERO process exit
code so an operator / scheduler treats a calibration breach as a real failure (SYS-02). Under
threshold it returns 0.

Reads the ledger via the injected ``bind``; the threshold via the injected ``settings``. Body lands
Wave 4; ``tests/test_drift.py`` pins the contract (RED).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

__all__ = ["monitor_drift"]


def monitor_drift(bind, settings, *, window_days, cities, models) -> int:
    """Trailing-window calibration-drift monitor → process exit code (SYS-02).

    Computes the trailing ``window_days`` ECE per (city, model) over the ledger and compares it to
    ``settings.drift_reliability_threshold`` (read from Settings, never hardcoded). Returns a
    NON-ZERO exit code and logs at ERROR on any breach, ``0`` when all strata are under threshold.
    Body lands Wave 4.
    """
    raise NotImplementedError("verify.drift.monitor_drift lands in Wave 4 (SYS-02).")
