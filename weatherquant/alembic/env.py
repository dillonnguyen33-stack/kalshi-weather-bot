"""Alembic migration environment.

Bound to the SQLAlchemy Core metadata so the initial ledger migration (plan 01-03)
can autogenerate (D-09). Two project-specific deviations from the stock scaffold:

1. The database URL is read from the ``DATABASE_URL`` env var (loaded from a local,
   git-ignored ``.env`` via python-dotenv) and MUST use the ``postgresql+psycopg://``
   dialect — psycopg v3, never psycopg2 (D-09 / Pitfall 4).

2. ``target_metadata`` imports ``weatherquant.db.models.metadata`` behind a
   forward-reference guard. The ``weatherquant.db`` package does not exist yet at
   plan 01-01 (it lands in 01-03); the try/except keeps this scaffold committable
   now and resolves automatically once the models module is written.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

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
    if "+psycopg" not in _database_url:
        raise RuntimeError(
            "DATABASE_URL must use the 'postgresql+psycopg://' dialect (psycopg v3), "
            f"not psycopg2. Got scheme in: {_database_url.split('://', 1)[0]}://"
        )
    config.set_main_option("sqlalchemy.url", _database_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here for 'autogenerate' support.
# Forward reference: weatherquant.db.models is created in plan 01-03. Until then the
# import fails and target_metadata stays None (autogenerate produces an empty diff,
# which is correct for 01-01 — there is no schema yet).
try:
    from weatherquant.db.models import metadata as target_metadata
except ImportError:
    target_metadata = None


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
