"""Shared pytest fixtures for the weatherquant test suite (plan 01-01 scaffold).

Two fixtures defined here:

* ``pg_engine`` — a SQLAlchemy ``Engine`` built from ``DATABASE_URL`` for the ledger
  integration tests (SYS-01). The URL is loaded from a local, git-ignored ``.env`` via
  python-dotenv so the suite runs under ``uv run pytest`` without the variable being
  pre-exported in the shell. If ``DATABASE_URL`` is genuinely unset, integration tests
  ``pytest.skip`` cleanly so the fast (no-DB) subset stays green. The scheme must be
  ``postgresql+psycopg://`` (psycopg v3 — D-09 / Pitfall 4); a wrong scheme fails loud.

* ``cli_fixture`` — loads the vendored NWS CLI parity fixtures (winter + summer obs and
  the CLI "Maximum" per city) used by ``test_cli_parity.py`` (D-04). Pure data; no
  network access, no live ingestion (ingestion is Phase 2).

NOTE: this module deliberately imports ``zoneinfo`` / ``timezonefinder`` nowhere on the
runtime path — those belong only in the test modules that derive/cross-check offsets
(D-02). conftest stays import-light.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

# Load DATABASE_URL (and any other vars) from a local .env so `uv run pytest` works
# without the operator pre-exporting the variable. dotenv does not override already-set
# environment variables, so an explicit shell export still wins.
try:
    from dotenv import load_dotenv

    # Resolve the package-root .env (weatherquant/.env), regardless of pytest's cwd.
    _ENV_PATH = pathlib.Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(_ENV_PATH)
    load_dotenv()  # also pick up a CWD .env if present
except ImportError:  # python-dotenv is a declared dep; guard for minimal envs only.
    pass


FIXTURE_DIR = pathlib.Path(__file__).resolve().parent / "fixtures" / "cli"


def _database_url() -> str | None:
    """Return DATABASE_URL from the environment, or None if genuinely unset/blank."""
    url = os.environ.get("DATABASE_URL")
    if url is not None and url.strip() == "":
        return None
    return url


@pytest.fixture(scope="session")
def pg_engine():
    """Yield a SQLAlchemy Engine bound to the test Postgres (DATABASE_URL).

    Skips cleanly (not errors) when DATABASE_URL is unset, so the fast subset stays
    green on machines without a database. When set, the engine is created and a trivial
    ``SELECT 1`` connectivity check runs; the schema itself is built by plan 01-03's
    Alembic migration / ``metadata.create_all`` (imported lazily so this fixture is
    collectable even before ``weatherquant.db`` exists).
    """
    url = _database_url()
    if url is None:
        pytest.skip(
            "DATABASE_URL unset — skipping ledger integration tests "
            "(set DATABASE_URL=postgresql+psycopg://... to enable)."
        )

    if "+psycopg" not in url:
        pytest.fail(
            "DATABASE_URL must use the 'postgresql+psycopg://' dialect (psycopg v3, "
            "not psycopg2) — D-09 / Pitfall 4."
        )

    import sqlalchemy as sa

    engine = sa.create_engine(url, future=True)

    # Connectivity check — proves Postgres is reachable before integration tests run.
    with engine.connect() as conn:
        conn.execute(sa.text("SELECT 1"))

    # Build the schema from the (about-to-exist) Core metadata. Imported lazily: at
    # plan 01-01 weatherquant.db.models does not exist yet, so the integration tests
    # are RED on this ImportError — correct, since the schema is delivered by 01-03.
    from weatherquant.db.models import metadata  # noqa: WPS433 (deferred import)

    metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


def _load_all_fixtures() -> dict:
    """Read every <city>.json under fixtures/cli/ into a dict keyed by city code."""
    data: dict = {}
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        payload = json.loads(path.read_text())
        data[payload["city"]] = payload
    return data


@pytest.fixture(scope="session")
def cli_fixture() -> dict:
    """Return the vendored NWS CLI parity fixtures keyed by Kalshi city code.

    Each entry: ``{"city", "station", "std_offset_hours", "days": {"winter"|"summer":
    {"date", "cli_max", "source_url", "obs": [{"ts_utc", "temp_f"}, ...]}}}``. The obs
    are UTC-timestamped hourly temperatures; the in-window maximum equals ``cli_max``,
    and each day includes at least one just-out-of-window hotter reading so the parity
    test proves the half-open window excludes the boundary (D-03 / D-04).
    """
    fixtures = _load_all_fixtures()
    if not fixtures:
        pytest.skip("No CLI fixtures found under tests/fixtures/cli/")
    return fixtures
