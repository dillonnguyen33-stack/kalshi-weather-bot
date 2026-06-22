"""Read helpers for the append-only ledger — the canonical "latest" idiom (D-10).

The "current truth" for a natural key is its row with the greatest ``available_at``,
tie-broken by greatest ``id`` (Pitfall 5), via PostgreSQL ``DISTINCT ON``:

    SELECT DISTINCT ON (<natural key>) *
    FROM <table>
    ORDER BY <natural key>, available_at DESC, id DESC;

Names resolve against real columns (``table.c[name]``), never f-string-interpolated
into SQL (threat T-01-05).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import sqlalchemy as sa
from sqlalchemy.engine import RowMapping

from weatherquant.db.models import NATURAL_KEYS, metadata
from weatherquant.db.types import Bind, exec_bind


def latest(
    bind: Bind,
    table_name: str,
    natural_key: Sequence[str] | None = None,
    where: Mapping[str, Any] | None = None,
) -> list[RowMapping]:
    """Return the newest row per natural key for ``table_name`` (DISTINCT ON, D-10).

    Args:
        bind: ``Engine`` or ``Connection`` to execute against.
        table_name: a ledger table in :data:`weatherquant.db.models.metadata`.
        natural_key: column names defining row identity. Defaults to the table's
            canonical key (:data:`weatherquant.db.models.NATURAL_KEYS`). If passed, must
            cover the full canonical key — a strict subset is rejected with ``ValueError``
            (an under-specified key silently collapses distinct facts, e.g. ensemble
            members). Resolved via ``table.c[name]``, never interpolated (T-01-05).
        where: optional ``{column: value}`` equality filter applied before the DISTINCT ON.
            Names resolve via ``table.c[name]``; values are bound params (T-01-05).

    Returns:
        One ``RowMapping`` per distinct natural-key tuple (greatest ``available_at``,
        ties by greatest ``id``), keyed by column name.
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
        # Reject a strict subset — omitted columns collapse distinct facts (WR-02 trap).
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

    with exec_bind(bind, write=False) as conn:
        return list(conn.execute(stmt).mappings().all())
