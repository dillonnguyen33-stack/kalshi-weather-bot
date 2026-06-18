"""RED stub — taker sweep + partial fill (PAP-03, 05-03 fills this GREEN).

A taker order walks resting levels best-first, partial-fills when liquidity is exhausted, and
reports the SIZE-WEIGHTED average price. Against ``scripted_book`` yes bids (50¢×100, 49¢×60)
a 130-contract taker sell fills 100@50 + 30@49 → avg ``(100*50 + 30*49)/130 = 49.77¢``; a
200-contract order partial-fills the 160 available. The ``-k sweep`` selector matches the
VALIDATION command ``pytest tests/test_fill_taker.py -k sweep``. ``fills.py`` is a PURE module.

Wave-0 RED stub: ``importorskip`` the not-yet-existing ``weatherquant.market.fills``.
"""

from __future__ import annotations

import pytest

fills = pytest.importorskip("weatherquant.market.fills")


@pytest.mark.xfail(reason="RED — 05-03 implements the taker sweep", strict=False)
def test_taker_sweep_size_weighted_avg_price(scripted_book):
    """A 130-contract sweep fills two levels at a size-weighted avg of 49.77¢."""
    raise NotImplementedError("05-03: walk levels best-first, size-weighted avg")


@pytest.mark.xfail(reason="RED — 05-03 implements the taker sweep", strict=False)
def test_taker_sweep_partial_fill_on_liquidity_exhaustion(scripted_book):
    """A 200-contract order partial-fills the 160 available, never fabricating liquidity."""
    raise NotImplementedError("05-03: partial fill = min(requested, available)")
