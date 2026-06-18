"""RED stub — fill stamped with real WS event time (PAP-03, D-08, 05-03 fills this GREEN).

A simulated fill's timestamp MUST be the real WS event time the trade occurred at — never
``datetime.now()`` and never back-dated. This is both a value assertion (the fill carries the
event time) AND a source-inspection guard (no ``datetime.now`` on the fill-building path),
mirroring ``tests/test_available_at.py`` (D-08, available_at = real event time).

Wave-0 RED stub: ``importorskip`` the not-yet-existing ``weatherquant.market.fills``.
"""

from __future__ import annotations

import pytest

fills = pytest.importorskip("weatherquant.market.fills")


@pytest.mark.xfail(reason="RED — 05-03 stamps fills with the WS event time", strict=False)
def test_fill_carries_ws_event_time_not_now():
    """The fill's event_time equals the WS message event time, not ``now()``."""
    raise NotImplementedError("05-03: fill.event_time = ws_event_time (passed in)")


@pytest.mark.xfail(reason="RED — 05-03 stamps fills with the WS event time", strict=False)
def test_no_datetime_now_on_fill_building_path():
    """Source-inspection: the fill-building path contains no ``datetime.now`` (D-08)."""
    raise NotImplementedError("05-03: AST/source scan asserts no datetime.now in fills.py")
