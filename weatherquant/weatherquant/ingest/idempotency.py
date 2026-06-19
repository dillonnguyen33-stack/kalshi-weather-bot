"""Content/cycle-aware skip-before-insert for the append-only ledger (D-10).

The ledger is insert-only: a per-table PostgreSQL trigger raises on any UPDATE / DELETE
(see :mod:`weatherquant.db.ddl`). So idempotency canNOT be an upsert or a blind re-insert
— it must be a **skip before INSERT** (RESEARCH Pitfall 6):

* Before writing, SELECT for an existing row matching the natural key AND the cycle AND the
  payload (content) columns. If such a row already exists, the write is a no-op.
* A genuinely new / corrected value (a changed payload) does NOT match, so it inserts a
  fresh row with a later ``available_at``. ``queries.latest()`` then returns the new value.

This means re-running an identical cycle is a true no-op (``COUNT(*)`` and ``latest()``
unchanged) while a correction appends — exactly the append-only contract, with the trigger
never provoked.

SQL-INJECTION SAFETY (T-02-03, ASVS V5). Column names from callers are resolved against
the table's real columns via ``table.c[name]`` (copying the ``queries.latest`` idiom);
nothing is ever f-string-interpolated into the SQL text. Values cross only as SQLAlchemy
Core bound parameters.
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
    """Build the equality/NULL predicate for one (column, value), JSONB-aware.

    A plain ``column == None`` renders ``column = NULL`` (never true), so a None must
    compare with ``IS NULL`` to match a stored SQL NULL. BUT a JSONB column stores a
    Python ``None`` as the JSON scalar ``'null'::jsonb`` (NOT SQL NULL), so ``IS NULL``
    would NOT match it — for JSONB we compare against the JSON-null literal instead. Both
    cases keep an identical re-run a true skip (D-10) without ever duplicating a row.
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
        bind: a SQLAlchemy ``Engine`` or ``Connection`` to execute against.
        table_name: name of a ledger table in :data:`weatherquant.db.models.metadata`
            (e.g. ``"forecasts"`` / ``"observations"``).
        natural_key_values: the natural-key columns -> values that define row identity
            (e.g. ``{"city": "NYC", "target_date": d, "model": "hrrr", "lead": 0,
            "member": 0, "cycle": c}``). Names are resolved via ``table.c[name]`` — never
            f-string-interpolated (T-02-03).
        content_cols: the payload columns -> values (e.g. ``{"temp_kelvin": 300.1, ...}``).
            A change in ANY content value makes the row "different", so a correction
            inserts a fresh row rather than matching an existing one (D-10).

    Returns:
        ``True`` if a row matching every natural-key AND content value is already present
        (the write should be skipped); ``False`` otherwise (proceed to INSERT).
    """
    table = metadata.tables[table_name]
    # Resolve names against the table's real columns — never f-string into SQL (T-02-03).
    # _match_condition handles the NULL / JSON-null subtleties so an identical re-run is a
    # true skip (D-10) and never a silent duplicate.
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


__all__ = ["row_exists", "already_ingested"]
