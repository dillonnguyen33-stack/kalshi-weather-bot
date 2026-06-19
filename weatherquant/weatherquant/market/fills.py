"""The pure paper-fill simulator (PAP-02/PAP-03) — taker sweep + maker queue, no real orders.

This is the heart of the Gate-1 fill realism. It is a PURE function over a book snapshot —
no ``websockets``/``cryptography``/SDK import — so the whole thing is unit-testable offline
with scripted books (it lives under ``market/`` only because it is the market edge's fill
accounting, not because it does I/O). Three correctness landmines are encoded here:

* **Pessimistic TAKER sweep (D-05/D-07).** :func:`taker_sweep` walks the *reflected* ask
  levels (produced by :func:`weatherquant.market.reflect.yes_ask_levels` /
  :func:`~weatherquant.market.reflect.no_ask_levels`) best-(cheapest-)price-first, taking
  ``min(available, remaining)`` at each level, and credits the SIZE-WEIGHTED average price.
  When liquidity runs out it credits a PARTIAL fill for what the book could actually absorb
  and records the shortfall — never an idealized single-price fill of the missing size.
  An empty ask side fills nothing and returns ``None`` (absence is absence, no fill row).
  The taker side is the Gate-1 credited path (D-05).

* **Conservative MAKER queue (D-06).** :func:`maker_queue_fill` joins BEHIND the size
  resting at our level at submit time and advances ONLY when a trade consumes size-ahead
  AND a taker crosses our level. Cancels-ahead (size removed without a crossing trade) do
  NOT advance us — counting them as progress over-credits maker fills in thin books
  (Pitfall 3). The maker model is the conservative shadow accounting (taker is credited).

* **WS-event-time stamping (D-08).** Every produced :class:`Fill` carries ``event_time`` =
  the real WS event time of the book state used — always a CALLER param, mirroring
  ``ingest/available_at.py``'s live-only-``now()`` fencing. There is NO ``datetime.now`` on
  the fill-producing path (enforced by ``tests/test_fill_event_time.py`` source inspection).

* **Structural EXECUTION_MODE guard (D-15).** :func:`assert_paper_mode` raises loudly if any
  order path is entered with ``execution_mode != 'live'``. There is simply NO REST
  order-submission code in this module (or anywhere under ``market/``) this milestone — the
  guard is the fence, the absence of an order endpoint is the structural property
  (``tests/test_no_live_orders.py``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Fill:
    """A single simulated fill credited against achievable book liquidity.

    Attributes:
        count: contracts actually filled (``> 0`` — a zero-fill is ``None``, never a ``Fill``).
        avg_price_cents: the SIZE-WEIGHTED average fill price in cents (D-07) — never an
            idealized single-price fill of unavailable liquidity.
        partial: ``True`` when the book could not absorb the full requested size; see
            :attr:`shortfall` for the unfilled remainder.
        shortfall: contracts that could NOT be filled (``want_count - count``); ``0`` for a
            full fill. Recorded so a partial is never silently treated as complete.
        is_maker: ``True`` for a maker (queue) fill, ``False`` for a taker (sweep) fill.
        event_time: the REAL WS event time of the book state used (D-08) — always a caller
            param, NEVER ``datetime.now()``/back-dated.
    """

    count: int
    avg_price_cents: float
    partial: bool
    shortfall: int
    is_maker: bool
    event_time: datetime


def _validate_levels(levels: Sequence[tuple[int, int]]) -> None:
    """Fail loud on a negative price or size in the level book (fail-loud input guard).

    Mirrors the ``price/``/``cli.run_price`` fail-loud idiom: an invalid book is a caller
    bug, not something to silently clamp. A non-negative ``(price, count)`` is required.
    """
    for price, count in levels:
        if price < 0 or count < 0:
            raise ValueError(
                f"ask level prices/sizes must be non-negative; got (price={price}, "
                f"count={count})"
            )


def taker_sweep(
    ask_levels: Sequence[tuple[int, int]],
    want_count: int,
    *,
    event_time: datetime,
    is_maker: bool = False,
) -> Fill | None:
    """Sweep ``want_count`` contracts against reflected ``ask_levels`` best-price-first (D-05/D-07).

    Walks the reflected ask levels (from :func:`weatherquant.market.reflect.yes_ask_levels` /
    :func:`~weatherquant.market.reflect.no_ask_levels`, already ordered cheapest-ask-first),
    taking ``take = min(available, want_count - filled)`` at each level and accumulating cost.
    The credited price is the SIZE-WEIGHTED average ``cost / filled`` (never an idealized
    single-price fill). When the book runs out before ``want_count`` it credits a PARTIAL fill
    for the achievable size and records the shortfall. An exhausted/empty book that fills
    nothing returns ``None`` — absence is absence (D-08), no fabricated fill row.

    Args:
        ask_levels: ``(price_cents, available_size)`` reflected ask levels, best-(cheapest-)
            first. Each must be non-negative (fail-loud otherwise).
        want_count: contracts to fill. Must be ``> 0`` (fail-loud otherwise).
        event_time: the real WS event time of the book state used — stamped onto the
            produced :class:`Fill` (D-08). NEVER ``datetime.now()``.
        is_maker: stamped onto the :class:`Fill`; ``False`` for a normal taker sweep.

    Returns:
        A :class:`Fill` crediting only achievable liquidity, or ``None`` if nothing filled.

    Raises:
        ValueError: if ``want_count`` is not positive, or a level price/size is negative.
    """
    if want_count <= 0:
        raise ValueError(f"want_count must be positive; got {want_count}")
    _validate_levels(ask_levels)

    filled = 0
    cost = 0
    for price, available in ask_levels:
        take = min(available, want_count - filled)
        filled += take
        cost += take * price
        if filled == want_count:
            break

    if filled == 0:
        # Absence is absence (D-08): an empty/exhausted book produces NO fill row, never a
        # fabricated fill of unavailable liquidity.
        return None

    return Fill(
        count=filled,
        avg_price_cents=cost / filled,
        partial=filled < want_count,
        shortfall=want_count - filled,
        is_maker=is_maker,
        event_time=event_time,
    )


# A book-change event kind on our resting level. ``trade`` is a genuine trade-through that
# consumes size (and, when it crosses our level, fills us); ``cancel`` is size removed WITHOUT
# a crossing trade — pessimistically it does NOT advance our queue position (D-06, Pitfall 3).
BookEventKind = Literal["trade", "cancel"]


@dataclass(frozen=True, slots=True)
class BookEvent:
    """One change observed at our maker level, carrying its real WS event time.

    Attributes:
        kind: ``"trade"`` (a genuine trade-through) or ``"cancel"`` (size removed without a
            crossing trade). Only a trade can consume size-ahead AND fill us; a cancel never
            advances the queue (the pessimistic D-06 default for the undocumented A3 mapping).
        size: the contract size of the change (``>= 0``).
        crosses_our_level: ``True`` when a trade prints at/through our resting price (a taker
            crossed our level). Required, in addition to consuming size-ahead, before any of
            our size fills. Cancels never cross.
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

    We join BEHIND ``size_ahead`` contracts resting at our level at submit time. Walking the
    ``events`` stream of changes at our level:

    * a ``cancel`` removes size-ahead from the book but does NOT advance our queue position —
      counting it as progress over-credits maker fills in thin books (the pessimistic default
      for the undocumented trade-vs-cancel mapping, A3);
    * a ``trade`` consumes size-ahead first; only once size-ahead is exhausted AND the trade
      ``crosses_our_level`` (a taker crossed our resting price) does the crossing remainder
      fill OUR size, up to ``our_size``.

    A market with NO trades (only cancels, or nothing) credits ZERO maker fills → ``None``
    (never the over-credit failure). The produced :class:`Fill` is ``is_maker=True`` and is
    stamped with the WS event time of the crossing trade that filled us (D-08).

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
            # Pessimistic D-06: a cancel removes size from the book but does NOT advance our
            # queue position. We deliberately do NOT decrement remaining_ahead here — counting
            # cancels as progress is the Pitfall-3 over-credit.
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
        # No crossing trade ever reached us — absence is absence (D-08), no maker fill row.
        return None

    return Fill(
        count=our_filled,
        # A maker rests AT its level, so every filled contract clears at our resting price;
        # the caller supplies that price out of band (the queue model proves the COUNT, the
        # conservative shadow accounting — taker is the credited Gate-1 path, D-05).
        avg_price_cents=0.0,
        partial=our_filled < our_size,
        shortfall=our_size - our_filled,
        is_maker=True,
        event_time=fill_event_time,
    )


@runtime_checkable
class _ExecutionModeSettings(Protocol):
    """Structural type for any object carrying a validated ``execution_mode`` (D-15)."""

    execution_mode: str


def assert_paper_mode(settings: _ExecutionModeSettings) -> None:
    """Structural EXECUTION_MODE no-live-orders guard (D-15) — fail loud outside ``live``.

    This is the fence that guards any (future, Gate-2) order-submission path: it is a no-op
    only when ``execution_mode == 'live'`` and RAISES loudly for anything else (``paper`` this
    milestone). Combined with the structural fact that NO REST order-submission code exists
    anywhere under ``market/`` this phase (asserted by ``tests/test_no_live_orders.py``), an
    accidental live order in paper mode is structurally unreachable (threat T-05-14).

    The guard is intentionally fail-CLOSED: it raises for ``paper`` (and any non-``live``
    value) so that wiring an order path without first flipping to validated ``live`` mode is a
    loud error, never a silent live submission. ``execution_mode`` itself is validated to the
    locked ``{paper, live}`` set at Settings construction (``db/engine.py``).

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
    "Fill",
    "BookEvent",
    "taker_sweep",
    "maker_queue_fill",
    "assert_paper_mode",
]
