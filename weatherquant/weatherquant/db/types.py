"""The single home for the SQLAlchemy execution-target type alias (CD-1/CD-2).

``Bind`` (``Engine | Connection``) is the one alias every read/write path uses to type
its ``bind`` parameter. Single-sourced HERE so duplicate definitions cannot drift.
Declared as an explicit ``TypeAlias`` (CD-2) so mypy treats it as a type regardless of
stub resolution.
"""

from __future__ import annotations

from typing import TypeAlias

from sqlalchemy.engine import Connection, Engine

# The ONE SQLAlchemy execution-target alias (CD-2 valid-type fix).
Bind: TypeAlias = Engine | Connection

__all__ = ["Bind"]
