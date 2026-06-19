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
from sqlalchemy.engine import Engine, RowMapping

from weatherquant.db.models import NATURAL_KEYS, metadata
from weatherquant.db.types import Bind


def latest(
    bind: Bind,
    table_name: str,
    natural_key: Sequence[str] | None = None,
    where: Mapping[str, Any] | None = None,
) -> list[RowMapping]:
    """Return the newest row per natural key for ``table_name`` (DISTINCT ON, D-10).

    Args:
        bind: a SQLAlchemy ``Engine`` or ``Connection`` to execute against.
        table_name: name of a ledger table defined in :data:`weatherquant.db.models
            .metadata` (e.g. ``"forecasts"``).
        natural_key: the natural-key column names defining row identity. Optional —
            defaults to the table's canonical key from
            :data:`weatherquant.db.models.NATURAL_KEYS`, which is the correct choice for
            almost every caller. If supplied explicitly it must cover the full canonical
            key: an under-specified key (a strict subset) DISTINCT-ONs over a narrower
            tuple and silently collapses distinct facts (e.g. two ensemble members) into
            one wrong "current truth", so it is rejected with ``ValueError``. Each name
            must be a real column; resolution goes through ``table.c[name]`` so no caller
            string is ever interpolated into the SQL text (T-01-05).
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

    canonical = NATURAL_KEYS.get(table_name)
    if natural_key is None:
        if canonical is None:
            raise ValueError(
                f"No canonical natural key registered for '{table_name}'; pass "
                f"natural_key explicitly (known tables: {sorted(NATURAL_KEYS)})."
            )
        natural_key = list(canonical)
    elif canonical is not None:
        # Reject a strict subset — the omitted columns would collapse distinct facts
        # into one row and return the wrong latest (the WR-02 ensemble-member trap).
        missing = [col for col in canonical if col not in set(natural_key)]
        if missing:
            raise ValueError(
                f"latest('{table_name}', ...) natural_key is missing key column(s) "
                f"{missing}; the full key is {list(canonical)}. An under-specified key "
                f"silently collapses distinct rows."
            )

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
