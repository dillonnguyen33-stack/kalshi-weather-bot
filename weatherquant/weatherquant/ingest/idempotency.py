"""Content/cycle-aware skip-before-insert for the append-only ledger (D-10).

The ledger is insert-only (a trigger raises on UPDATE/DELETE), so idempotency is a SELECT for
an existing row matching natural key + cycle + payload: a match is a no-op, a changed payload
appends a fresh row with a later ``available_at`` (Pitfall 6). Column names resolve via
``table.c[name]`` and values cross as bound params — never f-string-interpolated (T-02-03,
ASVS V5; see docs/DECISIONS.md).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

from weatherquant.db.models import metadata
from weatherquant.db.types import Bind


def _match_condition(column: sa.Column[Any], value: object) -> sa.ColumnElement[bool]:
    """Build the equality/NULL predicate for one (column, value), JSONB-aware (D-10).

    A None compares with ``IS NULL`` for a normal column, but a JSONB None is stored as
    ``'null'::jsonb`` (not SQL NULL), so JSONB compares against the JSON-null literal instead.
    """
    is_jsonb = isinstance(column.type, JSONB)
    if value is None:
        if is_jsonb:
            # Stored as 'null'::jsonb; match the JSON-null scalar, not SQL NULL.
            return column == sa.cast(sa.text("'null'"), JSONB)
        return column.is_(None)
    if is_jsonb:
        return column == sa.cast(sa.bindparam(None, value, type_=JSONB), JSONB)
    return column == value


def row_exists(
    bind: Bind,
    table_name: str,
    natural_key_values: Mapping[str, object],
    content_cols: Mapping[str, object],
) -> bool:
    """Return whether an identical ledger row already exists (skip-before-insert, D-10).

    Args:
        table_name: a ledger table in :data:`weatherquant.db.models.metadata`.
        natural_key_values: the columns defining row identity; names resolve via
            ``table.c[name]`` — never f-string-interpolated (T-02-03).
        content_cols: payload columns; a change in any value makes the row "different", so a
            correction inserts a fresh row rather than matching (D-10).

    Returns:
        ``True`` if a matching row exists (skip the write); ``False`` otherwise (INSERT).
    """
    table = metadata.tables[table_name]
    # Resolve names against real columns — never f-string into SQL (T-02-03).
    conditions = [
        _match_condition(table.c[name], value)
        for name, value in {**natural_key_values, **content_cols}.items()
    ]
    stmt = sa.select(sa.literal(1)).select_from(table).where(*conditions).limit(1)

    if isinstance(bind, Engine):
        with bind.connect() as conn:
            return conn.execute(stmt).first() is not None
    return bind.execute(stmt).first() is not None


# Test-facing alias (the Wave-0 RED stub imports ``already_ingested``). It IS the
# skip-before-insert check — a row already in the ledger means the cycle was already
# ingested, so the re-run skips (D-10). Kept as a thin alias so there is exactly ONE
# implementation of the idempotency predicate.
already_ingested = row_exists


__all__ = ["already_ingested", "row_exists"]
