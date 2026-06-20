"""The SINGLE point-in-time-of-knowledge helper (D-09) — the worst landmine in Phase 2.

The one place the codebase decides when a forecast datum became available. Backfill returns
``cycle_init + PUBLISH_LATENCY[model]``, never ``now()`` (Pitfall 5): too-late only withholds
data while too-early leaks, so the latencies are conservative lower bounds — err late, never
early. The live branch returns ``datetime.now(timezone.utc)`` and is the ONLY ``datetime.now``
in this module (the backfill branch must have none — enforced by source inspection in
tests/test_available_at.py). Observations do not use this helper; their ``available_at`` is the
feed's own report time (see docs/DECISIONS.md).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Literal

# Per-model publish latency after cycle init (RESEARCH Pitfall 5). Each value is
# cycle_init -> realistic availability, treated as a conservative lower bound: too-late
# only withholds data, too-early leaks (D-09). HRRR/GFS/GEFS are the HydroForecast figures
# (a ~30-min completeness buffer folded into each). NBM is the human-approved Wave-0 probe
# value recorded in 02-01-SUMMARY (PUBLISH_LATENCY[nbm] = 2h; observed ~1-2.34h, safe-late).
PUBLISH_LATENCY: dict[str, timedelta] = {
    "hrrr": timedelta(hours=1, minutes=30),   # ~1.5h (HydroForecast)
    "gfs": timedelta(hours=5, minutes=15),    # ~5.25h (HydroForecast)
    "gefs": timedelta(hours=6, minutes=30),   # ~6.5h (HydroForecast)
    "nbm": timedelta(hours=2),                # human-approved Wave-0 probe value (02-01)
}


def available_at(
    cycle_init: datetime,
    model: str,
    mode: Literal["backfill", "live"],
) -> datetime:
    """Return the point-in-time-of-knowledge for a forecast row (D-09).

    Args:
        cycle_init: the model run's init time (tz-aware UTC); the backfill latency anchor.
        model: NOAA model label; an unknown one raises ``KeyError`` (no silent default — ASVS V5).
        mode: ``"backfill"`` reconstructs historical availability; ``"live"`` returns now.

    Returns:
        A tz-aware UTC ``datetime``: backfill = ``cycle_init + PUBLISH_LATENCY[model]`` (never
        ``now()``); live = ``datetime.now(timezone.utc)``.

    Raises:
        KeyError: if ``model`` is not in :data:`PUBLISH_LATENCY` (backfill mode).
    """
    if mode == "live":
        # LIVE branch ONLY — the single sanctioned datetime.now in this module. When the
        # system actually held the decoded datum (D-09).
        return datetime.now(timezone.utc)
    # BACKFILL branch — NO datetime.now (Pitfall 5). Realistic historical availability =
    # the cycle's publish instant. KeyError on an unknown model, never a silent default.
    return cycle_init + PUBLISH_LATENCY[model]


__all__ = ["PUBLISH_LATENCY", "available_at"]
