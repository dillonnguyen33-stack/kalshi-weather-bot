"""Fill stamped with the real WS event time (PAP-03, D-08) — 05-03 GREEN.

A simulated fill's timestamp MUST be the real WS event time the trade occurred at — never
``datetime.now()`` and never back-dated. This is both a VALUE assertion (the fill carries the
passed-in event time) AND a SOURCE-inspection guard (no ``datetime.now`` anywhere on the
fill-building path), mirroring ``tests/test_available_at.py`` / ``test_no_runtime_dst.py``.
"""

from __future__ import annotations

import ast
import inspect
from datetime import datetime, timezone

from weatherquant.market import fills

WS_EVENT_TIME = datetime(2026, 6, 18, 19, 55, tzinfo=timezone.utc)


def test_fill_carries_ws_event_time_not_now():
    """The taker fill's event_time equals the passed-in WS event time, not ``now()``."""
    fill = fills.taker_sweep([(50, 100)], 50, event_time=WS_EVENT_TIME)
    assert fill is not None
    assert fill.event_time == WS_EVENT_TIME
    # It is exactly the caller's value — not a fresh wall-clock instant.
    assert fill.event_time != datetime.now(timezone.utc)


def test_no_datetime_now_on_fill_building_path():
    """Source-inspection: ``fills.py`` contains no ``datetime.now`` token anywhere (D-08).

    Unlike ``available_at`` (which fences a single sanctioned ``now()`` into the live branch),
    the pure fill simulator has NO sanctioned ``now()`` at all — ``event_time`` is always a
    caller param. Any ``datetime.now`` attribute access in the module is a violation.
    """
    source = inspect.getsource(fills)
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "now"
            and isinstance(node.value, ast.Name)
            and node.value.id == "datetime"
        ):
            raise AssertionError(
                "datetime.now found on the fill-building path (D-08 / Pitfall 4): the fill "
                "must carry the real WS event time, never the wall clock."
            )
