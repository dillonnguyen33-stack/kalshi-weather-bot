"""Correctness-alarm exception hierarchy for the ingestion spine (D-11, WR-05/NEW-1).

Correctness alarms (unit/sanity/window/rowcount breaches) must fail loud, unlike expected
transient failures that degrade gracefully (D-11). They are distinguished by type: every alarm
subclasses :class:`CorrectnessError`, so the orchestrator's one ``except CorrectnessError``
re-raise covers all current and future alarms (closing the WR-05 gap of bare ``ValueError``s).
Each concrete alarm ALSO subclasses :class:`ValueError` so existing ``pytest.raises(ValueError)``
call sites/tests keep working — a ``CorrectnessError`` IS-A ``ValueError`` (see docs/DECISIONS.md).
"""

from __future__ import annotations


class CorrectnessError(Exception):
    """Base class for ingestion correctness alarms that MUST fail loud (D-11, WR-05).

    Distinguished by type so the orchestrator re-raises these rather than swallowing them.
    """


class UnitError(CorrectnessError, ValueError):
    """A temperature unit is not Kelvin / cannot be converted (D-07, Pitfall 3).

    Also a ``ValueError`` so ``pytest.raises(ValueError, match="units must be 'K'")`` holds.
    """


class SanityError(CorrectnessError, ValueError):
    """A physical sanity contract was breached — lead-0 probe or snap-distance bound (D-04).

    Also a ``ValueError`` to preserve the lead-0 / snap-distance test contracts.
    """


class TargetDateError(CorrectnessError, ValueError):
    """No settlement window contains the valid instant — broken offset/window math (D-16, NEW-1).

    The WR-04 fail-loud raise (a silent UTC date would mislabel ``target_date``). Also a
    ``ValueError`` for call-site compatibility.
    """


class AvailabilityError(CorrectnessError, ValueError):
    """A point-in-time ``available_at`` cannot be derived honestly (D-09, CR-01).

    Raised when a backfill row would otherwise be stamped with the wall clock (the look-ahead
    leak). A correctness alarm (fail loud) and a ``ValueError`` for call-site compatibility.
    """


class CalibrationError(CorrectnessError, ValueError):
    """A calibration fit produced numbers that must not be priced on (CR-02).

    Raised on non-finite params/CRPS loss or an empty stratum — anything that would corrupt
    every downstream price. A money-path correctness alarm in the shared ``CorrectnessError``
    family so one fail-loud contract covers ingestion and calibration alike.
    """


class ObsFetchError(RuntimeError):
    """An ASOS/METAR obs fetch failed transiently (e.g. HTTP 429 after retries) — skip the day.

    Deliberately NOT a :class:`CorrectnessError`: a rate-limited fetch is an *expected transient*
    failure, not a correctness breach. Raising it (instead of returning ``[]``) makes the failure
    visible to the orchestrator so a fabricated empty label is never persisted; because it is a
    plain ``RuntimeError`` the orchestrator's generic degrade handler skips *this day* and
    continues the backfill, rather than aborting the whole run as a ``CorrectnessError`` would.
    The distinction it preserves: "fetch failed (unknown high)" vs. an HTTP-200 empty answer
    ("genuinely no obs", which still legitimately persists ``obs_count=0``).
    """


__all__ = [
    "AvailabilityError",
    "CalibrationError",
    "CorrectnessError",
    "ObsFetchError",
    "SanityError",
    "TargetDateError",
    "UnitError",
]
