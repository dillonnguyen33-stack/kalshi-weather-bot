"""Derived per-trade closing-line value (CLV) — PAP-04, pure, on the one LST clock.

CLV scores a fill against the volume-weighted CLOSING mid over the final window before
settlement. DERIVED post-settlement, NEVER a stamped fill column (D-12; see docs/DECISIONS.md):
a PURE function over the fill plus the closing-window ``market_snapshots`` rows.

Correctness landmines (see docs/DECISIONS.md):

* Settlement-clock anchor (D-10, the v3 founding bug): the half-open LST window
  ``[end_utc - CLV_WINDOW_MINUTES, end_utc)`` off :mod:`weatherquant.time`, never a re-derived
  civil-time clock.
* Volume-weighted mid (D-09): ``Σ(mid_i·vol_i)/Σvol_i`` in cents; empty/zero-volume fails loud.
* Side orientation (D-09 / WR-02): each contract is valued against its OWN side's closing mid —
  ``yes_mid - price`` for a ``"buy"`` (YES), ``(100 - yes_mid) - price`` for a ``"sell"`` (a NO
  buy whose ``avg_price_cents`` is NO-denominated, mirroring ``metrics.roi_from_fills``). NOT a
  bare sign flip against the YES mid (that differenced a NO price against a YES mid — a units bug).

PURE: imports only :func:`weatherquant.time.settlement_window`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any, Protocol

from weatherquant.registry import City
from weatherquant.time import coerce_utc, parse_utc, settlement_window

# Closing window length: the final N minutes before settlement (D-09). run_paper's snapshot
# cadence MUST stay strictly finer so the window holds >= 1 snapshot (PAP-04, T-05-20).
CLV_WINDOW_MINUTES = 30


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
            return coerce_utc(value)
    raw = snapshot.get("snapshot_for")
    if isinstance(raw, str):
        # Strip an optional ``#<seq>`` disambiguation suffix before parsing (WR-03).
        iso = raw.split("#", 1)[0]
        return parse_utc(iso)
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
    """Per-trade CLV in cents: a fill better than its OWN-side close is POSITIVE (D-09 / WR-02).

    Both operands are CENTS. The fill price is the un-rounded float ``avg_price_cents``, NEVER
    the rounded integer ``fills.price`` column (which re-introduces the +/-0.5c rounding bias,
    WR-05). :func:`vol_weighted_mid` returns the YES-side closing mid.

    SIDE ORIENTATION (WR-02 — the units fix). ``avg_price_cents`` is the price of the contract the
    fill actually BOUGHT, denominated on that side:

    * ``"buy"`` is a YES-contract buy; its closing value is the YES mid → ``edge = yes_mid - price``.
    * ``"sell"`` is a NO-contract buy (the ``side='no'`` ledger fill, normalized to ``"sell"`` by
      ``cli.verify._settle_window_fills``); its ``avg_price_cents`` is the NO contract's OWN price
      (mirroring ``metrics.roi_from_fills``, which buys NO at that price and pays 100c on a NO
      settlement). Its closing value is therefore the NO closing mid ``100 - yes_mid``, NOT the YES
      mid → ``edge = (100 - yes_mid) - price``. Bought the NO cheap relative to its close → POSITIVE.

    The previous orientation returned ``-(yes_mid - price)`` for a sell, differencing a NO price
    against a YES mid — a units mismatch that inflated/deflated the NO CLV by roughly the full
    0–100 span and inverted its sign. CLV is a Gate-1 money metric, so each contract is valued
    against its OWN side's closing mid.

    Args:
        fill: the executed fill (carries the size-weighted ``avg_price_cents`` on the bought side).
        closing_snapshots: the closing-window snapshots (typically the output of
            :func:`closing_window_snapshots`).
        side: ``"buy"`` (a YES buy) or ``"sell"`` (a NO buy) — selects the side's closing mid.

    Raises:
        ValueError: on an unknown ``side`` or an empty/zero-volume closing window.
    """
    if side not in ("buy", "sell"):
        raise ValueError(f"side must be 'buy' or 'sell'; got {side!r}")
    yes_mid = vol_weighted_mid(closing_snapshots)
    # Value each contract against its OWN side's closing mid: YES buy vs the YES mid, NO buy (the
    # "sell" alias) vs the NO mid (100 - yes_mid). avg_price_cents is already that side's own price.
    side_mid = yes_mid if side == "buy" else 100.0 - yes_mid
    return side_mid - fill.avg_price_cents


__all__ = [
    "CLV_WINDOW_MINUTES",
    "closing_window_snapshots",
    "clv_cents",
    "snapshot_event_time",
    "vol_weighted_mid",
]
