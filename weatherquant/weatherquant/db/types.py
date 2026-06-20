"""The single home for the SQLAlchemy execution-target type alias (CD-1/CD-2).

``Bind`` is the one alias every write/read path uses to type its ``bind`` parameter — a
SQLAlchemy ``Engine`` OR ``Connection``. It lives HERE, in exactly one module, so the
contract cannot drift: previously ``writer.py`` and ``persist.py`` each defined their own
``Bind = Engine | Connection`` (the writer comment even falsely claimed it was
"single-sourced"), and ``queries.py`` / ``idempotency.py`` / ``calibrate/strata.py`` spelled
the union inline. Two definitions can silently diverge (one adds ``AsyncConnection``, the
other does not), weakening the typed write surface — so there is now ONE definition.

It is declared as an explicit ``TypeAlias`` (CD-2): a bare ``Bind = Engine | Connection``
trips mypy's ``valid-type`` complaint when stub resolution is ambiguous, so spelling it
``Bind: TypeAlias = …`` makes mypy treat it as a type unconditionally.

A ``bind`` built by :func:`weatherquant.db.engine.get_engine` carries the preserve_rowcount
contract (D-11), so typing it consistently here keeps the one audited insert path honest.
"""

from __future__ import annotations

from typing import TypeAlias

from sqlalchemy.engine import Connection, Engine

# The ONE SQLAlchemy execution-target alias. Declared as an explicit TypeAlias so mypy
# treats it as a type regardless of stub resolution (CD-2 valid-type fix).
Bind: TypeAlias = Engine | Connection

__all__ = ["Bind"]
