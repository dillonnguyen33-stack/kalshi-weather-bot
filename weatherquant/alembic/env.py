"""Alembic migration environment.

Bound to the SQLAlchemy Core metadata so the initial ledger migration (plan 01-03)
can autogenerate (D-09). Two project-specific deviations from the stock scaffold:

1. The database URL is read from the ``DATABASE_URL`` env var (loaded from a local,
   git-ignored ``.env`` via python-dotenv) and MUST use the ``postgresql+psycopg://``
   dialect — psycopg v3, never psycopg2 (D-09 / Pitfall 4).

2. ``target_metadata`` is a HARD import of ``weatherquant.db.models.metadata``. A failed
   import (broken module) raises rather than silently leaving ``target_metadata=None``,
   which would make autogenerate emit an empty diff and hide schema drift.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# MetaData for 'autogenerate' support. This is a HARD import on purpose: if
# weatherquant.db.models cannot be imported (a real breakage — bad syntax, missing dep,
# broken DDL), autogenerate must fail loud rather than silently set target_metadata=None,
# which would emit an EMPTY diff and mask schema drift. (An earlier try/except guard was
# only a forward-reference workaround for when the models module did not yet exist; it
# does now, so the guard is removed.)
from weatherquant.db.models import metadata as target_metadata

# Load DATABASE_URL from a local .env if present (dev convenience; never committed).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is a declared dep; guard only for minimal envs.
    pass

import os

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Resolve the SQLAlchemy URL from the environment (DATABASE_URL) rather than from
# alembic.ini, so the real connection string is never committed. The scheme must be
# postgresql+psycopg:// (psycopg v3). If DATABASE_URL is unset we leave whatever the
# .ini provides — offline `--sql` generation does not require a live DB.
_database_url = os.environ.get("DATABASE_URL")
if _database_url:
    # Exact-match dialect guard, single-sourced in weatherquant.db.engine so the env,
    # the Settings validator, and the test conftest cannot diverge (Pitfall 4 / D-09).
    from weatherquant.db.engine import require_psycopg3_scheme

    require_psycopg3_scheme(_database_url)
    config.set_main_option("sqlalchemy.url", _database_url)

# Configure Python logging from the alembic.ini file, if one is in use.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
