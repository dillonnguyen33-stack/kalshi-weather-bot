"""The pure paper-fill simulator (PAP-02/PAP-03) — taker sweep + maker queue, no real orders.

A PURE function over a book snapshot (no I/O), unit-testable offline. Correctness landmines
(see docs/DECISIONS.md):

* Pessimistic TAKER sweep (D-05/D-07): :func:`taker_sweep` walks reflected ask levels
  cheapest-first, credits the SIZE-WEIGHTED average, and records a PARTIAL fill + shortfall
  when liquidity runs out; an empty book returns ``None``. The Gate-1 credited path.
* Conservative MAKER queue (D-06): :func:`maker_queue_fill` advances ONLY when a trade
  consumes size-ahead AND a taker crosses our level; cancels-ahead never advance us
  (Pitfall 3). Shadow accounting.
* WS-event-time stamping (D-08): every :class:`Fill` carries the book state's real WS event
  time as a CALLER param — the fill path never back-dates it (enforced by
  ``tests/test_fill_event_time.py`` source inspection).
* Structural EXECUTION_MODE guard (D-15): :func:`assert_paper_mode` fails loud outside
  ``live``; no order-submission code exists under ``market/`` this milestone
  (``tests/test_no_live_orders.py``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol


@dataclass(frozen=True, slots=True)
class Fill:
    """A single simulated fill credited against achievable book liquidity.

    Attributes:
        count: contracts actually filled (``> 0`` — a zero-fill is ``None``).
        avg_price_cents: the SIZE-WEIGHTED average fill price in cents (D-07).
        partial: ``True`` when the book could not absorb the full requested size.
        shortfall: contracts that could NOT be filled (``want_count - count``); ``0`` for full.
        is_maker: ``True`` for a maker (queue) fill, ``False`` for a taker (sweep) fill.
        event_time: the REAL WS event time of the book state used (D-08) — always a caller
            param, never back-dated.
    """

    count: int
    avg_price_cents: float
    partial: bool
    shortfall: int
    is_maker: bool
    event_time: datetime


def taker_sweep(
    ask_levels: Sequence[tuple[int, int]],
    want_count: int,
    *,
    event_time: datetime,
) -> Fill | None:
    """Sweep ``want_count`` contracts against reflected ``ask_levels`` best-price-first (D-05/D-07).

    Credits the SIZE-WEIGHTED average ``cost / filled``; a book that runs out before
    ``want_count`` credits a PARTIAL fill + shortfall, and an empty book returns ``None``.

    Args:
        ask_levels: ``(price_cents, available_size)`` reflected ask levels, cheapest-first.
            Each must be non-negative (fail-loud otherwise).
        want_count: contracts to fill. Must be ``> 0`` (fail-loud otherwise).
        event_time: the real WS event time of the book state, stamped onto the :class:`Fill`
            (D-08); a caller param, never back-dated.

    Returns:
        A :class:`Fill` crediting only achievable liquidity, or ``None`` if nothing filled.

    Raises:
        ValueError: if ``want_count`` is not positive, or a level price/size is negative.
    """
    if want_count <= 0:
        raise ValueError(f"want_count must be positive; got {want_count}")
    # Validate EVERY level up front (a negative price/size is a caller bug, never clamped) before
    # crediting any liquidity — preserves the prior _validate_levels "check all first" semantics.
    for price, count in ask_levels:
        if price < 0 or count < 0:
            raise ValueError(
                f"ask level prices/sizes must be non-negative; got (price={price}, count={count})"
            )

    filled = 0
    cost = 0
    for price, available in ask_levels:
        take = min(available, want_count - filled)
        filled += take
        cost += take * price
        if filled == want_count:
            break

    if filled == 0:
        # An empty/exhausted book produces NO fill row (D-08).
        return None

    return Fill(
        count=filled,
        avg_price_cents=cost / filled,
        partial=filled < want_count,
        shortfall=want_count - filled,
        is_maker=False,  # a taker sweep is never a maker fill (maker goes via maker_queue_fill)
        event_time=event_time,
    )


# A book-change event kind on our resting level: a ``trade`` consumes size (and fills us when
# it crosses); a ``cancel`` never advances our queue position (D-06, Pitfall 3).
BookEventKind = Literal["trade", "cancel"]


@dataclass(frozen=True, slots=True)
class BookEvent:
    """One change observed at our maker level, carrying its real WS event time.

    Attributes:
        kind: ``"trade"`` or ``"cancel"``; only a trade can fill us (D-06, A3).
        size: the contract size of the change (``>= 0``).
        crosses_our_level: ``True`` when a trade prints at/through our resting price; required,
            in addition to consuming size-ahead, before any of our size fills. Cancels never cross.
        event_time: the real WS event time of this change (D-08), stamped onto any fill.
    """

    kind: BookEventKind
    size: int
    crosses_our_level: bool
    event_time: datetime


def maker_queue_fill(
    size_ahead: int,
    our_size: int,
    events: Sequence[BookEvent],
) -> Fill | None:
    """Conservative maker queue fill: cancels-ahead never advance us (D-06, Pitfall 3).

    We join BEHIND ``size_ahead`` contracts at submit time. A ``cancel`` does NOT advance our
    queue position (A3); a ``trade`` consumes size-ahead first, then — once it is exhausted AND
    the trade ``crosses_our_level`` — the crossing remainder fills OUR size, up to ``our_size``.
    A market with no crossing trade returns ``None``. The :class:`Fill` is ``is_maker=True``,
    stamped with the WS event time of the crossing trade (D-08).

    Args:
        size_ahead: contracts resting ahead of us at submit time (``>= 0``).
        our_size: our resting order size (``> 0``).
        events: the ordered stream of changes at our level (trades and cancels).

    Returns:
        A maker :class:`Fill` for the crossed size (partial allowed), or ``None`` if no
        crossing trade ever reached us.

    Raises:
        ValueError: if ``our_size`` is not positive or ``size_ahead`` is negative.
    """
    if our_size <= 0:
        raise ValueError(f"our_size must be positive; got {our_size}")
    if size_ahead < 0:
        raise ValueError(f"size_ahead must be non-negative; got {size_ahead}")

    remaining_ahead = size_ahead
    our_filled = 0
    fill_event_time: datetime | None = None

    for event in events:
        if event.size < 0:
            raise ValueError(f"event size must be non-negative; got {event.size}")
        if event.kind == "cancel":
            # D-06: a cancel does NOT advance our queue position (Pitfall-3 over-credit).
            continue
        # A trade consumes size-ahead first.
        consumed_ahead = min(remaining_ahead, event.size)
        remaining_ahead -= consumed_ahead
        crossing_size = event.size - consumed_ahead
        # We fill ONLY when size-ahead is exhausted AND a taker crossed our level.
        if crossing_size > 0 and event.crosses_our_level:
            take = min(crossing_size, our_size - our_filled)
            if take > 0:
                our_filled += take
                fill_event_time = event.event_time
            if our_filled == our_size:
                break

    if our_filled == 0 or fill_event_time is None:
        # No crossing trade ever reached us — no maker fill row (D-08).
        return None

    return Fill(
        count=our_filled,
        # A maker clears at its resting price, supplied out of band (the queue model proves the
        # COUNT; taker is the credited Gate-1 path, D-05). The placeholder is NaN, not 0.0, so a
        # forgotten stamp fails loud: the writer rejects a non-finite maker price and CLV poisons
        # to NaN rather than silently reading closing_mid - 0 (CORR-MED-4; see docs/DECISIONS.md).
        avg_price_cents=math.nan,
        partial=our_filled < our_size,
        shortfall=our_size - our_filled,
        is_maker=True,
        event_time=fill_event_time,
    )


class _ExecutionModeSettings(Protocol):
    """Structural type for any object carrying a validated ``execution_mode`` (D-15)."""

    execution_mode: str


def assert_paper_mode(settings: _ExecutionModeSettings) -> None:
    """Structural EXECUTION_MODE no-live-orders guard (D-15) — fail loud outside ``live``.

    Fail-CLOSED: a no-op only at ``execution_mode == 'live'``, raises for anything else
    (``paper`` this milestone). With NO order-submission code under ``market/`` this phase
    (``tests/test_no_live_orders.py``), an accidental live order is unreachable (T-05-14).

    Args:
        settings: any object exposing a validated ``execution_mode`` string.

    Raises:
        RuntimeError: if ``execution_mode != 'live'`` — i.e. an order path was reached while
            not in validated live mode.
    """
    if settings.execution_mode != "live":
        raise RuntimeError(
            "refusing to enter an order-submission path with execution_mode="
            f"{settings.execution_mode!r}: live orders are unreachable until validated "
            "'live' mode (D-15, Gate 2). No order path exists this milestone."
        )


__all__ = [
    "BookEvent",
    "Fill",
    "assert_paper_mode",
    "maker_queue_fill",
    "taker_sweep",
]
