"""Correctness-alarm exception hierarchy for the ingestion spine (D-11, WR-05/NEW-1).

The graceful-degradation contract (D-11) says an EXPECTED transient failure — a late/missing
model cycle, an HTTP fetch error, a Herbie/cfgrib decode error — logs a structured fallback
and ingestion PROCEEDS with the other sources (absence = absence). But a CORRECTNESS ALARM is
the opposite: a unit mismatch, a lead-0 sanity breach, a snap-distance breach, an impossible
settlement window, or a single-insert rowcount breach all mean a real bug is firing and corrupt
data is at risk. Those must FAIL LOUD, never be downgraded to a silent "missing cycle" skip.

The two are distinguished by TYPE. Every correctness check raises a :class:`CorrectnessError`
subclass; the orchestrator's per-source ``except`` re-raises :class:`CorrectnessError` (and
``AssertionError``) and degrades gracefully on everything else. Re-raising the base class covers
all current and future alarms without enumerating each one at the catch site (the WR-05 gap was
catching ONLY ``WriteIntegrityError``/``AssertionError`` while the alarms below were bare
``ValueError``s — silently swallowed).

Each concrete alarm ALSO subclasses :class:`ValueError` so existing call sites and tests that
assert ``pytest.raises(ValueError)`` (the Kelvin-units guard, the snap-distance bound, the
lead-0 probe) keep their contract — a ``CorrectnessError`` IS-A ``ValueError`` while still being
catchable as the dedicated alarm base.
"""

from __future__ import annotations


class CorrectnessError(Exception):
    """Base class for ingestion correctness alarms that MUST fail loud (D-11, WR-05).

    Distinguished from transient/degradation failures by type so the orchestrator's
    graceful-degradation handler re-raises these rather than swallowing them as a skip.
    """


class UnitError(CorrectnessError, ValueError):
    """A temperature unit is not Kelvin / cannot be converted (D-07, Pitfall 3).

    Also a ``ValueError`` so existing ``pytest.raises(ValueError, match="units must be 'K'")``
    and the sources' unit-conversion contracts continue to hold.
    """


class SanityError(CorrectnessError, ValueError):
    """A physical sanity contract was breached — lead-0 probe or snap-distance bound (D-04).

    Also a ``ValueError`` to preserve the existing lead-0 / snap-distance test contracts.
    """


class TargetDateError(CorrectnessError, ValueError):
    """No settlement window contains the valid instant — broken offset/window math (D-16, NEW-1).

    The WR-04 fail-loud raise: a silent hand-rolled UTC date would mislabel ``target_date`` and
    the obs path would never join against it. Also a ``ValueError`` for call-site compatibility.
    """


class AvailabilityError(CorrectnessError, ValueError):
    """A point-in-time ``available_at`` cannot be derived honestly (D-09, CR-01).

    Raised when a backfill (historical) row would otherwise be stamped with the wall clock —
    the look-ahead leak the spine exists to prevent. A correctness alarm (fail loud) and a
    ``ValueError`` for call-site/test compatibility.
    """


class CalibrationError(CorrectnessError, ValueError):
    """A calibration fit produced numbers that must not be priced on (CR-02).

    Raised when the EMOS/NGR fit returns non-finite params, hits a non-finite CRPS loss, or is
    handed an empty stratum — anything that, if persisted, would silently corrupt every
    downstream price (``mu = a + b*m`` → NaN). A money-path correctness alarm: fail loud here,
    never write a corrupt ``calibration_params`` row. Lives in the shared ``CorrectnessError``
    family so one fail-loud contract covers both the ingestion spine and the calibration path.
    """


__all__ = [
    "CorrectnessError",
    "UnitError",
    "SanityError",
    "TargetDateError",
    "AvailabilityError",
    "CalibrationError",
]
