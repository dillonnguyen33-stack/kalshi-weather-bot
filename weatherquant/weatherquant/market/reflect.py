"""The ONE yes/no bid-only reflection seam (PAP-02, the central correctness landmine).

Kalshi's orderbook returns ONLY the bid side of each outcome — there are ``yes`` bids and
``no`` bids, but no native ask side. The ask you actually trade against is *reflected* from
the opposite outcome's bids: a ``no`` bid at price X is a ``yes`` ask at ``100 - X`` (cents),
and symmetrically a ``yes`` bid at X is a ``no`` ask at ``100 - X`` — each reflected level
carries the SIZE of the source bid level.

Getting this wrong silently inverts every fill price and CLV (Pitfall 1), so it lives in
exactly ONE place: every fill price/size computation in later plans (the taker sweep in
05-03, the maker model, CLV) routes through :func:`yes_ask_levels` / :func:`no_ask_levels`
rather than re-deriving the ``100 - price`` reflection.

This module is PURE (RESEARCH Pattern 3): no I/O, no ``websockets``/``cryptography``/SDK
imports — it mirrors the pure-NumPy ``price/`` boundary and stays a thin, offline-testable
seam. Output is ordered best-price-first for the synthesized side (cheapest yes-ask first),
so the 05-03 taker sweep can walk it directly.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# Kalshi prices are integer cents in 1..99; an outcome and its complement sum to 100¢.
_COMPLEMENT = 100


def _levels(book: object, side: str) -> Iterable[tuple[int, int]]:
    """Pull the ``(price_cents, count)`` bid levels for ``side`` from a book.

    Supports both a mapping/dict-shaped book (``book["yes"]`` — the conftest fixtures and
    the REST snapshot shape) and an attribute-shaped book (``book.yes`` — the in-memory
    ``Book`` object built in 05-02). Each level is a ``(price, count)`` pair (lists are
    accepted and coerced to tuples).
    """
    if isinstance(book, Mapping):
        raw = book[side]
    else:
        raw = getattr(book, side)
    return [(int(price), int(count)) for price, count in raw]


def _reflect(levels: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Reflect bid ``levels`` to the opposite ask side: ``(100 - price, count)``.

    Output is sorted cheapest-ask-first (best for a taker), which corresponds to the
    highest source-bid price first — preserving each level's count exactly.
    """
    # Highest bid price first -> lowest (100 - price) ask first == best/cheapest ask first.
    ordered = sorted(levels, key=lambda level: level[0], reverse=True)
    return [(_COMPLEMENT - price, count) for price, count in ordered]


def yes_ask_levels(book: object) -> list[tuple[int, int]]:
    """Return the synthesized YES ask levels reflected from the book's ``no`` bids.

    ``yes_ask = 100 - no_bid_price`` carrying the no-bid level's size, best (cheapest) yes
    ask first. An empty ``no`` side reflects to ``[]`` (absence is absence, never a
    fabricated level).

    Example: ``no`` bids ``[(40, 100), (38, 50)]`` -> yes asks ``[(60, 100), (62, 50)]``.
    """
    return _reflect(_levels(book, "no"))


def no_ask_levels(book: object) -> list[tuple[int, int]]:
    """Return the synthesized NO ask levels reflected from the book's ``yes`` bids.

    ``no_ask = 100 - yes_bid_price`` carrying the yes-bid level's size, best (cheapest) no
    ask first. An empty ``yes`` side reflects to ``[]``.

    Example: ``yes`` bids ``[(55, 30)]`` -> no asks ``[(45, 30)]``.
    """
    return _reflect(_levels(book, "yes"))


__all__ = ["yes_ask_levels", "no_ask_levels"]
