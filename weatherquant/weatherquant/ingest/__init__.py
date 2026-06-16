"""Ingestion package — the shared ingestion spine (Phase 2).

This package holds the four modules that define the interfaces every Wave's source
plan consumes (02-03 obs/AFD, 02-04 supplementary sources, 02-05 orchestration):

* :mod:`weatherquant.ingest.available_at` — the SINGLE point-in-time helper (D-09).
* :mod:`weatherquant.ingest.idempotency` — content/cycle skip-before-insert (D-10).
* :mod:`weatherquant.ingest.writer` — the SINGLE audited write path for both ledger
  payload tables (``insert_forecast`` + ``insert_observation``, D-11).
* :mod:`weatherquant.ingest.grib` — Herbie byte-range fetch + cfgrib decode +
  nearest-point snap + lead-0 probe (ING-01/02), converting to ``np.ndarray`` at the
  I/O edge (D-02).

Keeping these as one package guarantees ONE code path for ``available_at`` and
idempotency across every source — never a divergent hand-rolled insert downstream.
"""

from __future__ import annotations
