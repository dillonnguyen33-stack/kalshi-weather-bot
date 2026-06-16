"""RED stub (integration) — ING-08 / D-10: re-running an identical cycle is a no-op.

Turned GREEN by 02-03/02-05's ``weatherquant.ingest.idempotency``. Idempotency is a
SKIP-before-insert (never an UPDATE/upsert — the append-only trigger would raise, Pitfall
6). Marked ``integration``: skips cleanly when DATABASE_URL is unset. RED at import until
the ingest package exists.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_reingesting_identical_cycle_inserts_no_duplicate(pg_engine):
    # RED: weatherquant.ingest.idempotency lands in 02-03 (ImportError until then).
    from weatherquant.ingest.idempotency import already_ingested

    assert callable(already_ingested)
