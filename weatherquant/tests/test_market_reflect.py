"""RED stub — yes/no bid-only reflection (PAP-02, 05-03 fills this GREEN).

Kalshi quotes only the BID side of each outcome. The yes-ask is reflected from the opposite
no-bid: ``yes_ask = 100 - no_bid`` (cents) carrying the no-bid's SIZE, and symmetrically
``no_ask = 100 - yes_bid``. ``reflect.py`` is a PURE module (no websockets/SDK import) — it
lives under ``market/`` but the no-leak boundary keeps it free of I/O imports.

Wave-0 RED stub: ``importorskip`` the not-yet-existing ``weatherquant.market.reflect``; uses
the ``orderbook_snapshot`` conftest fixture (best no bid 49¢ → yes-ask 51¢).
"""

from __future__ import annotations

import pytest

reflect = pytest.importorskip("weatherquant.market.reflect")


@pytest.mark.xfail(reason="RED — 05-03 implements reflection", strict=False)
def test_yes_ask_is_hundred_minus_no_bid_with_size(orderbook_snapshot):
    """yes-ask = ``100 - best_no_bid`` (51¢) carrying the no-bid's size (80)."""
    raise NotImplementedError("05-03: yes_ask = 100 - no_bid, size = no_bid size")


@pytest.mark.xfail(reason="RED — 05-03 implements reflection", strict=False)
def test_no_ask_is_hundred_minus_yes_bid_with_size(orderbook_snapshot):
    """no-ask = ``100 - best_yes_bid`` (53¢) carrying the yes-bid's size (120)."""
    raise NotImplementedError("05-03: no_ask = 100 - yes_bid, size = yes_bid size")
