"""RED stub — no reachable order-submission path; EXECUTION_MODE guard (D-15, 05-03 GREEN).

A structural guard: there is NO reachable live order-submission path this milestone (Gate 1).
``execution_mode`` defaults to ``"paper"`` and gates whether any (future, Gate-2) live path is
reachable; an out-of-policy value cannot even construct (validated in ``db/engine.py``). This
test is the structural property check that the simulator never wires a real order submit while
in paper mode (a source-inspection / call-graph guard, like the no-``datetime.now`` guards).

Wave-0 RED stub: ``importorskip`` the not-yet-existing ``weatherquant.market`` package so the
structural assertions land once the package exists; 05-03 makes it GREEN.
"""

from __future__ import annotations

import pytest

market = pytest.importorskip("weatherquant.market")


@pytest.mark.xfail(reason="RED — 05-03 lands the structural no-order-path guard", strict=False)
def test_no_reachable_order_submission_path_in_paper_mode():
    """No live order-submission call is reachable while execution_mode is 'paper' (D-15)."""
    raise NotImplementedError("05-03: assert no order-submit symbol on the paper call graph")


@pytest.mark.xfail(reason="RED — 05-03 lands the structural no-order-path guard", strict=False)
def test_execution_mode_gates_any_future_live_path():
    """Any future live path is fenced behind an explicit execution_mode == 'live' check."""
    raise NotImplementedError("05-03: live path guarded by execution_mode, default paper")
