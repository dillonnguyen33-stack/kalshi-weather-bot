"""The ONE yes/no bid-only reflection seam (PAP-02, the central correctness landmine).

Kalshi quotes only the bid side of each outcome; the ask is reflected from the opposite
outcome's bids (``ask = 100 - opposite_bid`` cents, carrying the source level's size). Routing
every fill price/CLV through here avoids re-deriving the reflection and inverting prices
(Pitfall 1). PURE: no I/O. Output is cheapest-ask-first so the taker sweep walks it directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# Kalshi prices are integer cents in 1..99; an outcome and its complement sum to 100¢.
_COMPLEMENT = 100


def _levels(book: object, side: str) -> Iterable[tuple[int, int]]:
    """Pull the ``(price_cents, count)`` bid levels for ``side`` from a book.

    Supports both a mapping/dict-shaped book (``book["yes"]``) and an attribute-shaped book
    (``book.yes``). Lists are coerced to tuples.
    """
    if isinstance(book, Mapping):
        raw = book[side]
    else:
        raw = getattr(book, side)
    return [(int(price), int(count)) for price, count in raw]


def _reflect(levels: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Reflect bid ``levels`` to the opposite ask side: ``(100 - price, count)``, cheapest-first."""
    # Highest bid price first -> lowest (100 - price) ask first == best/cheapest ask first.
    ordered = sorted(levels, key=lambda level: level[0], reverse=True)
    return [(_COMPLEMENT - price, count) for price, count in ordered]


def best_bid(book: object, side: str) -> int | None:
    """Return the best (highest) bid PRICE in cents for ``side`` (``"yes"``/``"no"``), or ``None``.

    The single top-of-book BID accessor (IN-03). An empty side returns ``None``.
    """
    prices = [price for price, _ in _levels(book, side)]
    return max(prices) if prices else None


def yes_ask_levels(book: object) -> list[tuple[int, int]]:
    """Return the synthesized YES ask levels (``100 - no_bid``) reflected from the ``no`` bids.

    Cheapest yes-ask first; an empty ``no`` side reflects to ``[]``.

    Example: ``no`` bids ``[(40, 100), (38, 50)]`` -> yes asks ``[(60, 100), (62, 50)]``.
    """
    return _reflect(_levels(book, "no"))


def no_ask_levels(book: object) -> list[tuple[int, int]]:
    """Return the synthesized NO ask levels (``100 - yes_bid``) reflected from the ``yes`` bids.

    Cheapest no-ask first; an empty ``yes`` side reflects to ``[]``.

    Example: ``yes`` bids ``[(55, 30)]`` -> no asks ``[(45, 30)]``.
    """
    return _reflect(_levels(book, "yes"))


__all__ = ["best_bid", "no_ask_levels", "yes_ask_levels"]
