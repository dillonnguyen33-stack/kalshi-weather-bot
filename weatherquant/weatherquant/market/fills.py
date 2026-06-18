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


__all__ = ["Fill", "taker_sweep"]
