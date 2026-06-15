"""weatherquant.db — append-only, point-in-time Postgres ledger (SYS-01 / D-09–D-12).

The five skeletal ledger tables (`forecasts`, `observations`, `calibration_params`,
`market_snapshots`, `fills`) share one append-only contract: a surrogate serial `id`,
a stable natural key (indexed, NEVER unique), and an `available_at timestamptz` set to
point-in-time-of-knowledge. No UPDATE, no DELETE, ever (D-10) — corrections are new
inserts with a later `available_at`. Phases 2–6 extend payload columns via Alembic.

Submodules:
* ``models``  — SQLAlchemy Core ``metadata`` + the 5 ``Table`` defs.
* ``engine``  — pydantic-settings ``Settings`` (DATABASE_URL) + ``get_engine()`` on the
  ``postgresql+psycopg://`` dialect (psycopg v3, never psycopg2).
* ``queries`` — the reusable ``latest(...)`` DISTINCT ON helper.
"""

from __future__ import annotations

from weatherquant.db.models import metadata

__all__ = ["metadata"]
