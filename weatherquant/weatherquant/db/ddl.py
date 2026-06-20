"""Single source for the append-only ledger enforcement DDL (D-10).

A shared ``raise_append_only()`` PL/pgSQL function plus per-table triggers raise on any
UPDATE, DELETE, or TRUNCATE. The ONE place this DDL is written; both
:mod:`weatherquant.db.models` (via ``create_all`` events) and the Alembic migration
consume these strings, so the two schemas cannot drift.

Load-bearing notes:
* STATIC raise message, NO ``%``: ``sa.DDL`` runs ``statement % context`` (a literal
  ``%`` would need ``%%``) while ``op.execute`` takes the string verbatim — a fixed
  no-``%`` message is safe identically in both call sites.
* BEFORE triggers, not DO INSTEAD rewrite rules: a rewrite rule would suppress INSERT's
  rowcount/RETURNING tag; a BEFORE trigger leaves INSERT untouched.
* Separate per-statement TRUNCATE trigger: ``BEFORE UPDATE OR DELETE FOR EACH ROW`` does
  NOT fire on TRUNCATE, which would otherwise silently wipe the ledger.
"""

from __future__ import annotations

# The five append-only ledger tables (D-10). Explicit tuple for deterministic install order.
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

    Two per table: ``<table>_append_only`` (BEFORE UPDATE OR DELETE FOR EACH ROW) and
    ``<table>_no_truncate`` (BEFORE TRUNCATE FOR EACH STATEMENT — the per-row trigger
    does not catch TRUNCATE).
    """
    return (
        f'CREATE TRIGGER "{table}_append_only" '
        f'BEFORE UPDATE OR DELETE ON "{table}" '
        f"FOR EACH ROW EXECUTE FUNCTION raise_append_only();",
        f'CREATE TRIGGER "{table}_no_truncate" '
        f'BEFORE TRUNCATE ON "{table}" '
        f"FOR EACH STATEMENT EXECUTE FUNCTION raise_append_only();",
    )


def drop_trigger_sql(table: str) -> tuple[str, ...]:
    """Return the DROP TRIGGER statements inverse to :func:`create_trigger_sql`."""
    return (
        f'DROP TRIGGER IF EXISTS "{table}_append_only" ON "{table}";',
        f'DROP TRIGGER IF EXISTS "{table}_no_truncate" ON "{table}";',
    )


__all__ = [
    "LEDGER_TABLES",
    "CREATE_RAISE_FUNCTION_SQL",
    "DROP_RAISE_FUNCTION_SQL",
    "create_trigger_sql",
    "drop_trigger_sql",
]
