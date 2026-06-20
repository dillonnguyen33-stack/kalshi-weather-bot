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


__all__ = [
    "CorrectnessError",
    "UnitError",
    "SanityError",
    "TargetDateError",
    "AvailabilityError",
    "CalibrationError",
]
