"""weatherquant.db — append-only, point-in-time Postgres ledger (SYS-01 / D-09–D-12).

Five ledger tables share one append-only contract: surrogate serial `id`, an indexed
(never unique) natural key, and an `available_at timestamptz`. No UPDATE/DELETE (D-10);
corrections are new inserts with a later `available_at`.

Submodules:
* ``models``  — SQLAlchemy Core ``metadata`` + the 5 ``Table`` defs + ``NATURAL_KEYS``.
* ``engine``  — typed ``Settings`` + ``get_engine()`` (``postgresql+psycopg://``, D-09).
* ``queries`` — the ``latest(...)`` DISTINCT ON helper.
* ``ddl``     — single-sourced append-only enforcement DDL (function + triggers).
* ``types``   — the ``Bind`` execution-target type alias.
"""

from __future__ import annotations

from weatherquant.db.models import metadata

__all__ = ["metadata"]
