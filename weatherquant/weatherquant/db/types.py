"""The single home for the SQLAlchemy execution-target type alias (CD-1/CD-2).

``Bind`` (``Engine | Connection``) is the one alias every read/write path uses to type
its ``bind`` parameter. Single-sourced HERE so duplicate definitions cannot drift.
Declared as an explicit ``TypeAlias`` (CD-2) so mypy treats it as a type regardless of
stub resolution.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeAlias

from sqlalchemy.engine import Connection, Engine

# The ONE SQLAlchemy execution-target alias (CD-2 valid-type fix).
Bind: TypeAlias = Engine | Connection


@contextmanager
def exec_bind(bind: Bind, *, write: bool) -> Iterator[Connection]:
    """Yield a Connection for ``bind`` — the one read/write txn-dispatch every path shares.

    An owned ``Engine`` opens ``begin()`` (write) or ``connect()`` (read); a caller-supplied
    ``Connection`` is used as-is so the caller keeps owning its transaction.
    """
    if not isinstance(bind, Engine):
        yield bind  # caller-supplied Connection; it owns its transaction.
    elif write:
        with bind.begin() as conn:
            yield conn
    else:
        with bind.connect() as conn:
            yield conn


__all__ = ["Bind", "exec_bind"]
