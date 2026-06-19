"""Derived per-trade closing-line value (CLV) — PAP-04, pure, on the one LST clock.

CLV measures a fill against the volume-weighted CLOSING mid over the final window before the
market settles. It is a DERIVED post-settlement computation, NEVER a stamped column on the
fill row (D-12): the closing mid does not exist at fill time, and the ledger is append-only,
so a stamped CLV would back-date a value that depends on later data. Instead this module is a
PURE function over the fill plus the closing-window ``market_snapshots`` rows.

Three correctness landmines are encoded here:

* **Settlement-clock anchor (D-10, the v3 founding bug).** :func:`closing_window_snapshots`
  selects the final :data:`CLV_WINDOW_MINUTES` before ``settlement_window(city, date).end_utc``
  — the ONE fixed-offset LST clock in :mod:`weatherquant.time`, the half-open EXCLUSIVE end.
  The window is ``[end_utc - CLV_WINDOW_MINUTES, end_utc)`` (half-open, mirroring
  ``orchestrator._target_date_for``): a snapshot AT or after ``end_utc`` is EXCLUDED, and one
  before the window start is excluded. NEVER a re-derived civil-time clock.

* **Volume-weighted mid (D-09).** :func:`vol_weighted_mid` is ``Σ(mid_i·vol_i)/Σvol_i`` in
  cent space — the principled, reversible default. An empty window or zero total volume FAILS
  LOUD (raises) rather than returning a fabricated mid (no fillable closing data → no CLV,
  never a silent 0).

* **Sign orientation (D-09).** :func:`clv_cents` returns ``closing_mid - fill_price`` for a
  ``"buy"`` (positive when we paid LESS than the close — a good fill) and flips the sign for
  a ``"sell"``.

PURE: no ``websockets``/``cryptography``/SDK import; it imports only
:func:`weatherquant.time.settlement_window` for the clock anchor.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from typing import Any, Protocol, runtime_checkable

from weatherquant.registry import City
from weatherquant.time import settlement_window

# The closing window length: the final N minutes before settlement (D-09). A principled,
# reversible config CONSTANT — long enough to hold real closing liquidity, short enough that
# the mid reflects the genuine pre-settlement consensus rather than mid-session noise. The
# run_paper snapshot cadence (PAPER_SNAPSHOT_CADENCE_SECONDS) MUST stay strictly finer than
# this so the window holds >= 1 persisted snapshot (PAP-04 cadence sufficiency, T-05-20).
CLV_WINDOW_MINUTES = 30


@runtime_checkable
class _HasAvgPrice(Protocol):
    """Structural type for a fill carrying a size-weighted ``avg_price_cents`` (D-07)."""

    avg_price_cents: float


def snapshot_event_time(snapshot: Mapping[str, Any]) -> datetime:
    """Extract a snapshot's UTC event time, fail-loud on absence (absence = absence).

    The SINGLE public snapshot event-time parse seam (DD-1): both the CLV closing-window
    selector here AND ``cli._snapshot_event_time`` delegate to this one body so the D-08
    stamping/parsing cannot drift across modules. Prefers an explicit ``event_time`` /
    ``available_at`` ``datetime`` (the persisted row + ``fetch_snapshot`` stamp shape); falls
    back to parsing the ``snapshot_for`` ISO string (the fixture / ISO-stamp shape). A naive
    datetime is assumed UTC. A missing/unparseable time is a caller bug — raise, never silently
    drop the row from the window (which would skew the closing mid).
    """
    for key in ("event_time", "available_at"):
        value = snapshot.get(key)
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
    raw = snapshot.get("snapshot_for")
    if isinstance(raw, str):
        # Accept a trailing 'Z' (the fixture shape) as UTC.
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    raise ValueError(
        f"snapshot carries no usable event time (event_time/available_at/snapshot_for): "
        f"{snapshot!r}"
    )


# Private alias kept so any internal call site / external import of the old name still resolves;
# the PUBLIC ``snapshot_event_time`` is the single seam (DD-1).
_event_time = snapshot_event_time


def closing_window_snapshots(
    snapshots: Sequence[Mapping[str, Any]],
    city: City,
    day: date,
) -> list[Mapping[str, Any]]:
    """Select the snapshots in the half-open closing window on the LST clock (D-10).

    The window is ``[end_utc - CLV_WINDOW_MINUTES, end_utc)`` where ``end_utc`` is
    ``settlement_window(city, day).end_utc`` — the ONE fixed-offset LST clock, half-open
    EXCLUSIVE end (mirroring ``orchestrator._target_date_for``: ``start <= t < end``). A
    snapshot AT ``end_utc`` is EXCLUDED; one before the window start is excluded. NEVER a
    re-derived civil-time clock (the v3 founding bug, D-10).

    Args:
        snapshots: ``market_snapshots``-shaped rows (each must carry a usable event time:
            an ``event_time``/``available_at`` datetime or a ``snapshot_for`` ISO string).
        city: the settlement city (drives the fixed standard offset, never DST).
        day: the LST settlement date.

    Returns:
        The subset of ``snapshots`` whose event time falls in the half-open closing window,
        preserving input order. May be empty (the caller's ``vol_weighted_mid`` then fails
        loud — no fabricated mid).
    """
    win = settlement_window(city, day)
    window_start = win.end_utc - timedelta(minutes=CLV_WINDOW_MINUTES)
    return [
        snap
        for snap in snapshots
        if window_start <= snapshot_event_time(snap) < win.end_utc
    ]


def vol_weighted_mid(closing_snapshots: Sequence[Mapping[str, Any]]) -> float:
    """Volume-weighted closing mid ``Σ(mid_i·vol_i)/Σvol_i`` in CENTS (D-09), fail-loud on empty.

    Each snapshot must carry a ``mid`` in CENTS (the float-valued half-cent midpoint persisted
    by ``run_paper`` — unit-consistent with ``best_yes_bid``/``best_no_bid``/``avg_price_cents``,
    NOT [0,1] dollars, CR-01) and a real ``volume`` (the per-snapshot resting top-of-book
    liquidity in whole contracts persisted via the audited writer, WR-01). The returned mid is
    therefore in CENTS, so ``clv_cents`` subtracts ``avg_price_cents`` directly with no
    conversion. An empty window or a zero total volume RAISES (no fillable closing data → no
    CLV; never a fabricated mid/silent 0).

    Raises:
        ValueError: if ``closing_snapshots`` is empty or the total volume is non-positive.
    """
    if not closing_snapshots:
        raise ValueError(
            "no snapshots in the closing window — cannot derive a closing mid "
            "(no fillable closing data; never fabricate a mid)."
        )
    total_vol = 0.0
    weighted = 0.0
    for snap in closing_snapshots:
        mid = float(snap["mid"])
        vol = float(snap["volume"])
        if vol < 0:
            raise ValueError(f"snapshot volume must be non-negative; got {vol}")
        weighted += mid * vol
        total_vol += vol
    if total_vol <= 0:
        raise ValueError(
            "total closing-window volume is zero — cannot derive a volume-weighted mid."
        )
    return weighted / total_vol


def clv_cents(
    fill: _HasAvgPrice,
    closing_snapshots: Sequence[Mapping[str, Any]],
    side: str,
) -> float:
    """Per-trade CLV in cents: a fill better than the close is POSITIVE (D-09).

    Both operands are CENTS: the closing mid (``vol_weighted_mid`` over snapshots whose ``mid``
    is float-valued cents, CR-01) and the fill's ``avg_price_cents``. ``edge = closing_mid -
    fill.avg_price_cents``; returns ``edge`` for a ``"buy"`` (positive when we paid LESS than the
    volume-weighted closing mid — a good fill) and ``-edge`` for a ``"sell"`` (the sign flips:
    selling ABOVE the close is the good fill). No unit conversion is needed now that the
    persisted ``mid`` is cents. The closing mid is derived from ``closing_snapshots`` via
    :func:`vol_weighted_mid` (which fails loud on an empty/zero-volume window — no fabricated CLV).

    Args:
        fill: the executed fill (carries the size-weighted ``avg_price_cents``).
        closing_snapshots: the closing-window snapshots (typically the output of
            :func:`closing_window_snapshots`).
        side: ``"buy"`` or ``"sell"`` — orients the sign.

    Raises:
        ValueError: on an unknown ``side`` or an empty/zero-volume closing window.
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell'; got {side!r}")
    closing_mid = vol_weighted_mid(closing_snapshots)
    edge = closing_mid - fill.avg_price_cents
    return edge if side == "buy" else -edge


__all__ = [
    "CLV_WINDOW_MINUTES",
    "snapshot_event_time",
    "closing_window_snapshots",
    "vol_weighted_mid",
    "clv_cents",
]
