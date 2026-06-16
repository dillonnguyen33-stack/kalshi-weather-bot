"""ING-09 / D-09: available_at = cycle_init + latency (backfill) vs now (live).

GREEN by 02-02's ``weatherquant.ingest.available_at``. This is the worst look-ahead
landmine in the phase (Pitfall 5): backfill MUST stamp ``cycle_init +
PUBLISH_LATENCY[model]`` and NEVER ``now()``; live stamps ``now()``. The NBM latency is
the human-approved Wave-0 probe value recorded in 02-01 (2h).

Table-driven over models x {backfill, live}, plus a source-inspection guard that the
backfill branch contains no ``datetime.now`` (mirrors the ``tests/test_no_runtime_dst.py``
source-guard style).
"""

from __future__ import annotations

import ast
import inspect
from datetime import datetime, timedelta, timezone

import pytest

from weatherquant.ingest.available_at import PUBLISH_LATENCY, available_at

_CYCLE = datetime(2026, 6, 13, 12, 0, tzinfo=timezone.utc)


def test_publish_latency_has_all_four_noaa_models():
    # The four NOAA-family models the GRIB core decodes (D-12). nbm is the Wave-0 value.
    assert set(PUBLISH_LATENCY) == {"hrrr", "gfs", "gefs", "nbm"}
    assert PUBLISH_LATENCY["nbm"] == timedelta(hours=2)  # human-approved 02-01 probe value


@pytest.mark.parametrize("model", ["hrrr", "gfs", "gefs", "nbm"])
def test_backfill_uses_cycle_plus_latency_not_now(model):
    # Backfill is deterministic: cycle_init + the model's publish latency, never now().
    backfilled = available_at(_CYCLE, model, mode="backfill")
    assert backfilled == _CYCLE + PUBLISH_LATENCY[model]


@pytest.mark.parametrize("model", ["hrrr", "gfs", "gefs", "nbm"])
def test_live_is_now_utc_tz_aware(model):
    before = datetime.now(timezone.utc)
    got = available_at(_CYCLE, model, mode="live")
    after = datetime.now(timezone.utc)
    # Live returns a tz-aware UTC instant within the call window (a few seconds of now()).
    assert got.tzinfo is not None
    assert got.utcoffset() == timedelta(0)
    assert before <= got <= after


def test_unknown_model_raises_keyerror_not_silent_default():
    with pytest.raises(KeyError):
        available_at(_CYCLE, "ecmwf", mode="backfill")


def test_backfill_branch_has_no_datetime_now_in_source():
    """Source guard (Pitfall 5): no ``datetime.now`` outside the live branch.

    Walk the AST of ``available_at``; the ONLY ``datetime.now`` call must sit inside the
    ``if mode == "live":`` branch. Any ``datetime.now`` reachable in the backfill path
    would silently break Phase-6 no-look-ahead, so this is asserted structurally.
    """
    source = inspect.getsource(available_at)
    tree = ast.parse(source)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)

    # Collect the live-branch node ids so we can exclude their datetime.now calls.
    live_branch_nodes: set[int] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.If):
            test = node.test
            is_live = (
                isinstance(test, ast.Compare)
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "live"
            )
            if is_live:
                for body_node in node.body:
                    for inner in ast.walk(body_node):
                        live_branch_nodes.add(id(inner))

    # Any datetime.now attribute access NOT inside the live branch is a violation.
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "now"
            and id(node) not in live_branch_nodes
        ):
            pytest.fail("datetime.now reachable outside the live branch (Pitfall 5)")
