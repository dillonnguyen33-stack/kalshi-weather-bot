"""Read helpers for the append-only ledger — the canonical "latest" idiom (D-10).

The ledger is insert-only: a single natural key may have many rows, each a successive
point-in-time observation of the same fact (D-12). The "current truth" for a key is the
row with the greatest ``available_at`` — with ``id`` as the deterministic tiebreaker
when two rows share an ``available_at`` (Pitfall 5). This is expressed once, here, via
PostgreSQL ``DISTINCT ON``:

    SELECT DISTINCT ON (<natural key>) *
    FROM <table>
    ORDER BY <natural key>, available_at DESC, id DESC;

Phases 2–6 reuse :func:`latest` rather than re-deriving the idiom. The query is built
entirely from SQLAlchemy Core constructs and validated column handles — natural-key
names are resolved against the table's real columns (``table.c[name]``), never
f-string-interpolated into SQL (threat T-01-05 / SQL injection via dynamic identifiers).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import Connection, Engine, RowMapping

from weatherquant.db.models import metadata


def latest(
    bind: Engine | Connection,
    table_name: str,
    natural_key: Sequence[str],
    where: Mapping[str, Any] | None = None,
) -> list[RowMapping]:
    """Return the newest row per natural key for ``table_name`` (DISTINCT ON, D-10).

    Args:
        bind: a SQLAlchemy ``Engine`` or ``Connection`` to execute against.
        table_name: name of a ledger table defined in :data:`weatherquant.db.models
            .metadata` (e.g. ``"forecasts"``).
        natural_key: the natural-key column names defining row identity (e.g.
            ``["city", "target_date", "model", "lead"]``). Each must be a real column of
            the table; resolution goes through ``table.c[name]`` so no caller string is
            ever interpolated into the SQL text (T-01-05).
        where: optional ``{column_name: value}`` equality filter applied BEFORE the
            DISTINCT ON, so callers can scope to a single key (e.g.
            ``{"city": "NYC", "model": "gfs"}``) instead of fetching every key and
            filtering in Python. Column names resolve through ``table.c[name]`` (same
            injection-safe path as ``natural_key``); values are passed as bound
            parameters by SQLAlchemy, never interpolated.

    Returns:
        One ``RowMapping`` per distinct natural-key tuple — the row with the greatest
        ``available_at``, breaking ties by greatest ``id``. Each mapping is keyed by
        column name (``row["city"]``, ``row["available_at"]``).

    The mandatory ordering is ``(*key_cols, available_at DESC, id DESC)``: the leading
    key columns satisfy ``DISTINCT ON``, ``available_at DESC`` selects the most recent
    point-in-time, and ``id DESC`` makes the choice deterministic under a tie (Pitfall 5).
    """
    table = metadata.tables[table_name]
    # Resolve names against the table's real columns — never f-string into SQL.
    key_cols = [table.c[name] for name in natural_key]

    stmt = (
        sa.select(table)
        .distinct(*key_cols)
        .order_by(*key_cols, table.c.available_at.desc(), table.c.id.desc())
    )

    if where:
        # Same validated-handle resolution as the key columns; values are bound params.
        stmt = stmt.where(*(table.c[name] == value for name, value in where.items()))

    if isinstance(bind, Engine):
        with bind.connect() as conn:
            return list(conn.execute(stmt).mappings().all())
    return list(bind.execute(stmt).mappings().all())
