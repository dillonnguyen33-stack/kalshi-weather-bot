"""GREEN — the ONE yes/no bid-only reflection seam (PAP-02, Pitfall 1).

Kalshi quotes only the BID side of each outcome. The yes-ask is reflected from the opposite
no-bid: ``yes_ask = 100 - no_bid`` (cents) carrying the no-bid's SIZE, and symmetrically
``no_ask = 100 - yes_bid``. ``reflect.py`` is a PURE module (no websockets/cryptography/SDK
import). These tests pin the reflection against the conftest ``orderbook_snapshot`` fixture
(best no bid 49¢/80 -> yes-ask 51¢/80; best yes bid 47¢/120 -> no-ask 53¢/120) plus inline
known levels for ordering and the empty-side case.
"""

from __future__ import annotations

import ast
import pathlib

from weatherquant.market import reflect
from weatherquant.market.book import OrderBook, apply


def test_yes_ask_is_hundred_minus_no_bid_with_size(orderbook_snapshot):
    """yes-ask = ``100 - best_no_bid`` (51¢) carrying the no-bid's size (80), best-first.

    The conftest ``orderbook_snapshot`` is now the VERIFIED enveloped WS message (05-11), so it
    is parsed into an ``OrderBook`` first (the reflect seam reflects a parsed book, not the raw
    wire envelope) — the cent-space top-of-book is unchanged, so the numbers are identical.
    """
    book = OrderBook()
    apply(book, orderbook_snapshot)
    asks = reflect.yes_ask_levels(book)
    # no bids [[49,80],[48,150],[47,260]] -> yes asks [(51,80),(52,150),(53,260)], cheapest first.
    assert asks == [(51, 80), (52, 150), (53, 260)]
    # Best (cheapest) yes ask is the reflection of the BEST (highest) no bid.
    assert asks[0] == (51, 80)


def test_no_ask_is_hundred_minus_yes_bid_with_size(orderbook_snapshot):
    """no-ask = ``100 - best_yes_bid`` (53¢) carrying the yes-bid's size (120), best-first.

    Parses the enveloped ``orderbook_snapshot`` into an ``OrderBook`` first (05-11); the cent-
    space top-of-book is unchanged, so the reflected numbers are identical.
    """
    book = OrderBook()
    apply(book, orderbook_snapshot)
    asks = reflect.no_ask_levels(book)
    # yes bids [[47,120],[46,200],[45,350]] -> no asks [(53,120),(54,200),(55,350)].
    assert asks == [(53, 120), (54, 200), (55, 350)]
    assert asks[0] == (53, 120)


def test_reflection_is_exact_hundred_minus_price_preserving_count():
    """Reflection is exactly ``100 - price`` in cent space with the per-level count carried."""
    book = {"no": [(40, 100), (38, 50)], "yes": [(55, 30)]}
    assert reflect.yes_ask_levels(book) == [(60, 100), (62, 50)]
    assert reflect.no_ask_levels(book) == [(45, 30)]


def test_output_is_best_price_first_regardless_of_input_order():
    """Unsorted input is reflected best-(cheapest)-ask-first for the synthesized side."""
    book = {"no": [(38, 50), (40, 100), (39, 70)], "yes": []}
    # Highest no bid (40) -> cheapest yes ask (60) first.
    assert reflect.yes_ask_levels(book) == [(60, 100), (61, 70), (62, 50)]


def test_empty_side_reflects_to_empty_list():
    """An empty bid side reflects to ``[]`` (absence, not a fabricated level)."""
    book = {"yes": [], "no": []}
    assert reflect.yes_ask_levels(book) == []
    assert reflect.no_ask_levels(book) == []


def test_attribute_shaped_book_is_supported():
    """An attribute-shaped book (the 05-02 Book object) reflects identically to a dict."""

    class _Book:
        yes = [(47, 120)]
        no = [(49, 80)]

    assert reflect.yes_ask_levels(_Book()) == [(51, 80)]
    assert reflect.no_ask_levels(_Book()) == [(53, 120)]


def test_reflect_module_imports_no_io_dependencies():
    """``reflect.py`` stays pure: no websockets/cryptography/SDK import (single pure seam)."""
    source = pathlib.Path(reflect.__file__).read_text()
    tree = ast.parse(source)
    forbidden = {"websockets", "cryptography"}
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert forbidden.isdisjoint(imported_roots), (
        f"reflect.py must not import {forbidden & imported_roots} (pure seam)"
    )
