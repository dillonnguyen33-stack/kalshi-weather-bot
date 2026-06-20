"""Derived per-trade closing-line value (CLV) — PAP-04, pure, on the one LST clock.

CLV scores a fill against the volume-weighted CLOSING mid over the final window before
settlement. DERIVED post-settlement, NEVER a stamped fill column (D-12; see docs/DECISIONS.md):
a PURE function over the fill plus the closing-window ``market_snapshots`` rows.

Correctness landmines (see docs/DECISIONS.md):

* Settlement-clock anchor (D-10, the v3 founding bug): the half-open LST window
  ``[end_utc - CLV_WINDOW_MINUTES, end_utc)`` off :mod:`weatherquant.time`, never a re-derived
  civil-time clock.
* Volume-weighted mid (D-09): ``Σ(mid_i·vol_i)/Σvol_i`` in cents; empty/zero-volume fails loud.
* Sign orientation (D-09): ``closing_mid - fill_price`` for a ``"buy"``, flipped for a ``"sell"``.

PURE: imports only :func:`weatherquant.time.settlement_window`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, UTC
from typing import Any, Protocol, runtime_checkable

from weatherquant.registry import City
from weatherquant.time import settlement_window

# Closing window length: the final N minutes before settlement (D-09). run_paper's snapshot
# cadence MUST stay strictly finer so the window holds >= 1 snapshot (PAP-04, T-05-20).
CLV_WINDOW_MINUTES = 30


@runtime_checkable
class _HasAvgPrice(Protocol):
    """Structural type for a fill carrying a size-weighted ``avg_price_cents`` (D-07)."""

    avg_price_cents: float


def snapshot_event_time(snapshot: Mapping[str, Any]) -> datetime:
    """Extract a snapshot's UTC event time, fail-loud on absence (DD-1, IN-01).

    The SINGLE public snapshot event-time parse seam (``cli._snapshot_event_time`` delegates
    here too, so D-08 stamping cannot drift). Preference order ``event_time`` then
    ``available_at`` then ``snapshot_for`` ISO string is a contract, not an accident — the
    closing-window axis is ``available_at`` (the observed book instant); run_paper sets all
    three from that instant and asserts agreement at persist time. Naive datetimes are UTC;
    a missing/unparseable time raises (never silently drop a row and skew the mid).
    """
    for key in ("event_time", "available_at"):
        value = snapshot.get(key)
        if isinstance(value, datetime):
            return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    raw = snapshot.get("snapshot_for")
    if isinstance(raw, str):
        # Strip an optional ``#<seq>`` disambiguation suffix before parsing (WR-03).
        iso = raw.split("#", 1)[0]
        # Accept a trailing 'Z' (the fixture shape) as UTC.
        parsed = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    raise ValueError(
        f"snapshot carries no usable event time (event_time/available_at/snapshot_for): "
        f"{snapshot!r}"
    )


def closing_window_snapshots(
    snapshots: Sequence[Mapping[str, Any]],
    city: City,
    day: date,
) -> list[Mapping[str, Any]]:
    """Select the snapshots in the half-open closing window on the LST clock (D-10).

    Window ``[end_utc - CLV_WINDOW_MINUTES, end_utc)`` off ``settlement_window(city, day)``;
    a snapshot AT ``end_utc`` is EXCLUDED.

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

    Each snapshot carries a ``mid`` in CENTS (float-valued, CR-01) and a ``volume`` that is the
    top-of-book size SUPPORTING this mid — ``min`` of best-yes-bid and reflected best-yes-ask
    size — NOT the two-sided union depth (05-06 MD-01; see docs/DECISIONS.md). Returned mid is
    in cents, so ``clv_cents`` subtracts ``avg_price_cents`` directly. Empty/zero-volume raises.

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

    Both operands are CENTS. The fill price is the un-rounded float ``avg_price_cents``, NEVER
    the rounded integer ``fills.price`` column (which re-introduces the +/-0.5c rounding bias,
    WR-05). ``edge = closing_mid - fill.avg_price_cents``; returns ``edge`` for a ``"buy"`` and
    ``-edge`` for a ``"sell"``. The closing mid comes from :func:`vol_weighted_mid`.

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
    "closing_window_snapshots",
    "clv_cents",
    "snapshot_event_time",
    "vol_weighted_mid",
]
