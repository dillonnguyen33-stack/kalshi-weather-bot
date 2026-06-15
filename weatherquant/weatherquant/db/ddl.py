"""Single source for the append-only ledger enforcement DDL (D-10).

The append-only guarantee is enforced structurally by PostgreSQL: a shared
``raise_append_only()`` PL/pgSQL function plus per-table triggers that raise on any
UPDATE, DELETE, or TRUNCATE. This module is the ONE place that DDL is written; both
:mod:`weatherquant.db.models` (via ``metadata.create_all`` DDL events) and the Alembic
migration consume the strings produced here, so the migrated schema and the
``create_all`` schema are guaranteed identical — there is no hand-duplicated SQL to
drift apart.

Why a STATIC raise message (no ``%`` placeholder): ``sqlalchemy.DDL`` runs
``statement % context`` for substitution, so a literal ``%`` in a ``sa.DDL`` body has to
be escaped as ``%%``, while ``op.execute`` in a migration takes the string verbatim.
Keeping the RAISE message a fixed string with NO ``%`` eliminates that escaping
divergence entirely — the exact same SQL string is safe in both call sites.

Why BEFORE triggers (not DO INSTEAD rewrite rules): a rewrite rule on a table suppresses
the row-count / RETURNING tag of OTHER commands (an INSERT would then report
``rowcount -1``), whereas a BEFORE trigger leaves INSERT semantics untouched and still
rejects the forbidden mutations with a clear exception.
"""

from __future__ import annotations

# The five append-only ledger tables (D-10). Kept as an explicit tuple so the install
# order is deterministic and both consumers iterate the identical set.
LEDGER_TABLES: tuple[str, ...] = (
    "forecasts",
    "observations",
    "calibration_params",
    "market_snapshots",
    "fills",
)

# Static RAISE message — NO ``%`` placeholder (see module docstring). Identical text is
# safe under both ``sa.DDL`` (% substitution) and ``op.execute`` (verbatim).
CREATE_RAISE_FUNCTION_SQL: str = (
    "CREATE OR REPLACE FUNCTION raise_append_only() RETURNS trigger AS $$ "
    "BEGIN RAISE EXCEPTION 'weatherquant ledger tables are append-only "
    "(D-10) — correct via a new INSERT with a later available_at'; "
    "END; $$ LANGUAGE plpgsql;"
)

DROP_RAISE_FUNCTION_SQL: str = "DROP FUNCTION IF EXISTS raise_append_only();"


def create_trigger_sql(table: str) -> tuple[str, ...]:
    """Return the CREATE TRIGGER statements that guard ``table`` (D-10).

    * ``<table>_append_only`` — ``BEFORE UPDATE OR DELETE ... FOR EACH ROW``: rejects
      row-level mutations while leaving INSERT semantics untouched.
    """
    return (
        f'CREATE TRIGGER "{table}_append_only" '
        f'BEFORE UPDATE OR DELETE ON "{table}" '
        f"FOR EACH ROW EXECUTE FUNCTION raise_append_only();",
    )


def drop_trigger_sql(table: str) -> tuple[str, ...]:
    """Return the DROP TRIGGER statements inverse to :func:`create_trigger_sql`."""
    return (
        f'DROP TRIGGER IF EXISTS "{table}_append_only" ON "{table}";',
    )


__all__ = [
    "LEDGER_TABLES",
    "CREATE_RAISE_FUNCTION_SQL",
    "DROP_RAISE_FUNCTION_SQL",
    "create_trigger_sql",
    "drop_trigger_sql",
]
