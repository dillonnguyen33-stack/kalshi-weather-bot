"""RED stub — insert_market_snapshot / insert_fill audited write path (D-13, 05-04 GREEN).

The two new audited writers extend ``weatherquant.ingest.writer`` and reuse its ``_insert_row``
helper verbatim: skip-before-insert (``row_exists``) returns a no-op on a duplicate natural-key
+ content row (idempotency, D-10), and an explicit ``rowcount != 1`` raise
(``WriteIntegrityError``, survives ``python -O``, D-11). ``available_at`` is ALWAYS a caller
param (= the real WS event time, never ``now()``). There is NO UPDATE/upsert path — a
correction is a later-``available_at`` INSERT.

Marked ``integration`` (the assertions exercise the DB via ``pg_conn``). Wave-0 RED stub:
``importorskip`` the not-yet-existing writer symbols so collection succeeds; 05-04 implements
``insert_market_snapshot`` / ``insert_fill`` and flips these GREEN.
"""

from __future__ import annotations

import pytest

writer = pytest.importorskip("weatherquant.ingest.writer")

pytestmark = pytest.mark.integration

_HAS_MARKET_WRITERS = hasattr(writer, "insert_market_snapshot") and hasattr(
    writer, "insert_fill"
)
_skip_until_0504 = pytest.mark.skipif(
    not _HAS_MARKET_WRITERS,
    reason="RED — 05-04 adds insert_market_snapshot / insert_fill to the writer",
)


@_skip_until_0504
def test_insert_market_snapshot_rowcount_one(pg_conn):
    """A single snapshot insert lands exactly one row (rowcount==1 contract, D-11)."""
    raise NotImplementedError("05-04: insert_market_snapshot delegates to _insert_row")


@_skip_until_0504
def test_insert_market_snapshot_idempotent(pg_conn):
    """Re-inserting an identical snapshot is a no-op skip, never a duplicate (D-10)."""
    raise NotImplementedError("05-04: row_exists skip-before-insert returns 0")


@_skip_until_0504
def test_insert_fill_rowcount_one(pg_conn):
    """A single fill insert lands exactly one row (rowcount==1 contract, D-11)."""
    raise NotImplementedError("05-04: insert_fill delegates to _insert_row")


@_skip_until_0504
def test_insert_fill_idempotent(pg_conn):
    """Re-inserting an identical fill is a no-op skip, never a duplicate (D-10)."""
    raise NotImplementedError("05-04: row_exists skip-before-insert returns 0")
