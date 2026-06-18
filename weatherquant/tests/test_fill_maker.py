"""RED stub — pessimistic maker queue (PAP-02, 05-03 fills this GREEN).

The maker queue is modeled PESSIMISTICALLY: cancels ahead in the queue do NOT advance our
position (only a genuine trade-through does), and a maker order fills ONLY when a trade occurs
AND the price crosses our resting level. Against ``scripted_book`` (size-ahead 100 at yes
50¢), a cancel of 40 ahead leaves us still behind 60; only a 100+-contract trade at/through
50¢ fills us. ``fills.py`` is a PURE module.

Wave-0 RED stub: ``importorskip`` the not-yet-existing ``weatherquant.market.fills``.
"""

from __future__ import annotations

import pytest

fills = pytest.importorskip("weatherquant.market.fills")


@pytest.mark.xfail(reason="RED — 05-03 implements the maker queue", strict=False)
def test_cancels_ahead_do_not_advance_queue(scripted_book):
    """A cancel ahead does NOT advance our queue position (pessimistic, PAP-02)."""
    raise NotImplementedError("05-03: only trade-through consumes size ahead, not cancels")


@pytest.mark.xfail(reason="RED — 05-03 implements the maker queue", strict=False)
def test_maker_fill_only_on_trade_and_cross(scripted_book):
    """A maker order fills only when a trade occurs AND it crosses our resting level."""
    raise NotImplementedError("05-03: fill iff trade size > size_ahead and price crosses")
