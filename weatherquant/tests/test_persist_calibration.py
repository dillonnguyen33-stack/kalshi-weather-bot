"""Persistence contract for store_calibration_params (D-13 / SYS-01).

Two layers:

* **Offline guard (no DB).** The single-row integrity guard is an EXPLICIT raise of
  ``WriteIntegrityError`` (a ``RuntimeError``), never a bare ``assert`` that ``python -O``
  strips (WR-06). A tiny fake Connection drives the non-1 rowcount branch offline, and an AST
  walk of the function proves there is no bare ``assert`` in the guard.
* **Integration (DATABASE_URL).** A real INSERT lands exactly one ``calibration_params`` row
  through the append-only path, a SECOND fit for the SAME natural key appends a NEW row (never
  an UPDATE), and ``latest()`` returns the most-recently-available params (D-13).
"""

from __future__ import annotations

import ast
import inspect
from datetime import date, datetime, timezone

import pytest

from weatherquant.calibrate.persist import store_calibration_params
from weatherquant.ingest.writer import WriteIntegrityError


def _payload(**overrides: object) -> dict:
    base = dict(
        city="NYC",
        model="hrrr",
        lead=24,
        month=6,
        mean_intercept=1.0,
        mean_slope=0.95,
        var_intercept=1.5,
        var_slope=0.8,
        sigma_floor=0.5,
        n_train=120,
        pool_level="month",
        crps_train=1.10,
        crps_oos=1.20,
        crps_baseline_oos=1.40,
        trained_through=date(2024, 6, 30),
        available_at=datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc),
    )
    base.update(overrides)
    return base


# --- Offline guard (no DB) ---------------------------------------------------------------


class _FakeResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _FakeConn:
    """A Connection stand-in (NOT an Engine) returning a controllable insert rowcount."""

    def __init__(self, rowcount: int) -> None:
        self._rowcount = rowcount

    def execute(self, _stmt):  # noqa: ANN001
        return _FakeResult(self._rowcount)


def test_rowcount_one_returns_one():
    conn = _FakeConn(1)
    assert store_calibration_params(conn, **_payload()) == 1  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [0, 2])
def test_non_one_rowcount_raises_write_integrity_error(bad: int):
    conn = _FakeConn(bad)
    with pytest.raises(WriteIntegrityError, match="expected rowcount==1"):
        store_calibration_params(conn, **_payload())  # type: ignore[arg-type]
    assert issubclass(WriteIntegrityError, RuntimeError)


def test_guard_is_not_a_bare_assert_survives_dash_O():
    """The rowcount guard is an ``if ... raise``, never a bare assert (WR-06, survives -O)."""
    tree = ast.parse(inspect.getsource(store_calibration_params))
    asserts = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    raises = [n for n in ast.walk(tree) if isinstance(n, ast.Raise)]
    assert not asserts, "the rowcount guard must not be a bare assert (stripped under -O)"
    assert raises, "the rowcount guard must raise an explicit exception"


def test_available_at_is_a_parameter_not_now():
    """``available_at`` is a function parameter — never computed via ``now()`` inside (D-13)."""
    sig = inspect.signature(store_calibration_params)
    assert "available_at" in sig.parameters
    # The function body must not CALL datetime.now()/utcnow() — the caller supplies the instant.
    # AST-scan for call expressions (so docstring prose mentioning ``now()`` is not a false hit).
    tree = ast.parse(inspect.getsource(store_calibration_params))
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    } | {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "now" not in called and "utcnow" not in called


# --- Integration (DATABASE_URL) ----------------------------------------------------------

pytestmark_integration = pytest.mark.integration


@pytest.mark.integration
def test_insert_lands_one_row_and_refit_appends(pg_conn):
    """A real INSERT lands one row; a refit for the SAME key APPENDS a new row (D-13)."""
    from weatherquant.db import queries

    key = dict(city="NYC", model="hrrr", lead=24, month=6)

    n1 = store_calibration_params(
        pg_conn,
        **_payload(
            **key,
            mean_intercept=1.0,
            available_at=datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc),
        ),
    )
    assert n1 == 1

    # A SECOND fit for the same natural key with a LATER available_at — append, never UPDATE.
    n2 = store_calibration_params(
        pg_conn,
        **_payload(
            **key,
            mean_intercept=2.0,
            available_at=datetime(2026, 6, 17, 12, 0, tzinfo=timezone.utc),
        ),
    )
    assert n2 == 1

    rows = queries.latest(pg_conn, "calibration_params", where=key)
    assert len(rows) == 1  # one row per natural key
    # latest() returns the MOST-RECENTLY-AVAILABLE params (the refit), proving append-only.
    assert rows[0]["mean_intercept"] == 2.0
    assert rows[0]["pool_level"] == "month"
