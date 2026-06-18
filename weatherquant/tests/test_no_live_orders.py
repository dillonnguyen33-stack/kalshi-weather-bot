"""No reachable order-submission path; structural EXECUTION_MODE guard (D-15) — 05-03 GREEN.

A structural property check: there is NO reachable live order-submission path this milestone
(Gate 1). Every Python module under ``weatherquant/market/`` is AST/source-scanned and must
contain NO order-submission token (no ``POST .../orders``, ``create_order``, ``place_order``,
``submit_order``) — the guard is the fence and the ABSENCE of an order endpoint is the
structural property (mirrors the ``test_no_runtime_dst.py`` / no-``datetime.now`` source-scan
idiom). Separately, :func:`assert_paper_mode` is fail-closed: a no-op only in validated
``live`` mode and RAISES for ``paper`` (and any other value), so wiring an order path without
first flipping to ``live`` is a loud error (threat T-05-14).
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass

import pytest

import weatherquant.market as market_pkg
from weatherquant.market import fills

# Order-submission tokens that MUST NOT appear anywhere under market/ this milestone. A REST
# order submit is a POST to an /orders endpoint or any *_order verb; none exist (Gate 1).
_ORDER_SUBMISSION_PATTERNS = [
    re.compile(r"/orders\b"),
    re.compile(r"\bcreate_order\b"),
    re.compile(r"\bplace_order\b"),
    re.compile(r"\bsubmit_order\b"),
    re.compile(r"\bcreate_market_order\b"),
    re.compile(r"""POST\s*["'].*orders""", re.IGNORECASE),
]


def _market_source_files() -> list[pathlib.Path]:
    pkg_dir = pathlib.Path(market_pkg.__file__).parent
    return sorted(p for p in pkg_dir.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_reachable_order_submission_path_in_paper_mode():
    """A source scan of weatherquant/market/ finds NO order-submission path (D-15)."""
    files = _market_source_files()
    assert files, "expected at least one module under weatherquant/market/"
    offenders: list[str] = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        for pattern in _ORDER_SUBMISSION_PATTERNS:
            if pattern.search(source):
                offenders.append(f"{path.name}: {pattern.pattern}")
    assert not offenders, (
        "order-submission path found under market/ (D-15 / T-05-14 — no live order code "
        f"exists this milestone): {offenders}"
    )


def test_execution_mode_gates_any_future_live_path():
    """assert_paper_mode is fail-closed: no-op only in 'live', raises for 'paper'/anything else."""

    @dataclass
    class _Settings:
        execution_mode: str

    # Paper (this milestone) — any order path is fenced: the guard RAISES loudly.
    with pytest.raises(RuntimeError, match="order-submission path"):
        fills.assert_paper_mode(_Settings(execution_mode="paper"))

    # Any non-live value likewise raises (fail-closed) — a typo can never unlock an order path.
    with pytest.raises(RuntimeError):
        fills.assert_paper_mode(_Settings(execution_mode="sandbox"))

    # Only validated live mode is a no-op (returns None, no raise) — the future Gate-2 path.
    assert fills.assert_paper_mode(_Settings(execution_mode="live")) is None
