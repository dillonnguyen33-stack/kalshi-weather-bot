"""The SINGLE point-in-time-of-knowledge helper (D-09) — the worst landmine in Phase 2.

``available_at(cycle_init, model, mode)`` is the ONE place the codebase decides *when a
datum became available to the system*. Every forecast write routes its ``available_at``
through here so the backfill-vs-live distinction lives in exactly one place.

THE LANDMINE (RESEARCH Pitfall 5, D-09). For a *backfill* the answer is NOT the wall
clock — it is ``cycle_init + PUBLISH_LATENCY[model]`` (the realistic instant the model
run actually became downloadable). Stamping a backfilled row with ``datetime.now()``
makes every historical forecast appear to have been "unavailable" until the backfill ran,
which silently destroys Phase 6's no-look-ahead walk-forward backtest. The inverse error
(stamping the nominal cycle time, with zero latency) leaks future information. The
asymmetry is deliberate and load-bearing: a slightly **too-late** ``available_at`` is
SAFE (it only withholds data); a **too-early** one LEAKS. So the latency table below is a
set of conservative lower bounds with a completeness buffer — err late, never early.

For a *live* fetch the answer IS ``datetime.now(timezone.utc)`` — the moment the running
system actually held the decoded datum. That ``now()`` call appears in this module's LIVE
branch ONLY; the backfill branch must contain no ``datetime.now`` reference (enforced by
``tests/test_available_at.py`` via source inspection).

OBSERVATIONS DO NOT USE THIS HELPER. An observation's ``available_at`` is its report /
availability time (set by the obs path in 02-03 from the feed's own timestamp), NOT
``now()`` and NOT the LST settlement-window edge. This helper governs the FORECAST point-
in-time only; mixing the obs report time in here would conflate two different clocks.
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
        cycle_init: the model run's init time (tz-aware UTC). Used only in the backfill
            branch as the anchor for ``cycle_init + PUBLISH_LATENCY[model]``.
        model: the NOAA model label — one of ``"hrrr"``, ``"gfs"``, ``"gefs"``, ``"nbm"``.
            An unknown model raises ``KeyError`` (never a silent default — ASVS V5).
        mode: ``"backfill"`` reconstructs the realistic historical availability;
            ``"live"`` returns the current instant the running system held the datum.

    Returns:
        A tz-aware UTC ``datetime``. For ``"backfill"``: ``cycle_init +
        PUBLISH_LATENCY[model]`` (deterministic, NEVER ``now()``). For ``"live"``:
        ``datetime.now(timezone.utc)``.

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
