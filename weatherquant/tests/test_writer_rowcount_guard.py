"""WR-06 / D-11: the single-row insert integrity guard is a real raise, not a bare assert.

The append-only single-write contract (D-11) requires that exactly one row land per audited
insert. v1 enforced this with ``assert result.rowcount == 1`` — but ``python -O`` /
PYTHONOPTIMIZE strips asserts, silently disabling the only check that the insert actually
wrote a row. These tests pin that the guard:

* raises a real exception type (``RuntimeError``, NOT ``AssertionError``) on a non-1 rowcount;
* survives ``-O`` — i.e. the guard's source is an ``if ... raise``, never a bare ``assert``.

No DB needed: a tiny fake Connection returns a controllable ``rowcount`` so the guard's
branch is exercised offline (mirrors the offline-stub style of test_graceful_degradation).
"""

from __future__ import annotations

import ast
import inspect
from datetime import date, datetime, timezone

import pytest
import sqlalchemy as sa
from sqlalchemy.engine import Connection

from weatherquant.ingest import writer
from weatherquant.ingest.writer import _insert_row


class _FakeResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _FakeConn:
    """Minimal stand-in for a SQLAlchemy Connection that returns a fixed insert rowcount.

    Spoofs ``isinstance(bind, Engine)`` to False (it is not an Engine) so ``_insert_row``
    treats it as a Connection and calls ``conn.execute`` directly — returning a result whose
    ``rowcount`` we control to drive the guard's non-1 branch.
    """

    def __init__(self, rowcount: int) -> None:
        self._rowcount = rowcount

    def execute(self, _stmt):  # noqa: ANN001
        return _FakeResult(self._rowcount)


# A throwaway in-memory table so table.insert() builds without a real schema/DB.
_TABLE = sa.Table(
    "guard_probe",
    sa.MetaData(),
    sa.Column("city", sa.String),
    sa.Column("target_date", sa.Date),
    sa.Column("available_at", sa.DateTime(timezone=True)),
    sa.Column("temp_kelvin", sa.Float),
)


def _call_with_rowcount(monkeypatch: pytest.MonkeyPatch, rowcount: int) -> int:
    # row_exists must say "no identical row" so _insert_row proceeds to the insert + guard.
    monkeypatch.setattr(writer, "row_exists", lambda *a, **k: False)
    # _insert_row branches on isinstance(bind, Engine); a _FakeConn is not an Engine, so it is
    # treated as a Connection and execute() is called directly (no bind.begin()).
    bind: Connection = _FakeConn(rowcount)  # type: ignore[assignment]
    return _insert_row(
        bind,
        _TABLE,
        {"city": "NYC", "target_date": date(2026, 6, 13)},
        {"temp_kelvin": 300.0},
        datetime(2026, 6, 13, 12, tzinfo=timezone.utc),
    )


def test_rowcount_one_returns_one(monkeypatch: pytest.MonkeyPatch):
    # The happy path: a single-row insert reports rowcount==1 and is accepted.
    assert _call_with_rowcount(monkeypatch, 1) == 1


@pytest.mark.parametrize("bad", [0, 2])
def test_non_one_rowcount_raises_runtimeerror_not_assertionerror(
    monkeypatch: pytest.MonkeyPatch, bad: int
):
    # A 0- or 2-row outcome is an integrity breach: it must raise a REAL exception type
    # (RuntimeError), not an AssertionError that -O would strip.
    with pytest.raises(RuntimeError, match="expected rowcount==1"):
        _call_with_rowcount(monkeypatch, bad)


def test_guard_is_not_a_bare_assert_survives_dash_O():
    """Source guard: the rowcount integrity check is an ``if ... raise``, never a bare assert.

    Walk the AST of ``_insert_row``; assert there is NO ``assert`` statement (which ``-O``
    strips) and that a ``raise`` exists. This structurally proves the D-11 single-insert guard
    survives ``python -O`` / PYTHONOPTIMIZE (WR-06).
    """
    source = inspect.getsource(_insert_row)
    tree = ast.parse(source)
    asserts = [n for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    raises = [n for n in ast.walk(tree) if isinstance(n, ast.Raise)]
    assert not asserts, "the rowcount guard must not be a bare assert (stripped under -O)"
    assert raises, "the rowcount guard must raise an explicit exception"
