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

02-05 adds :mod:`weatherquant.ingest.orchestrator`, which COMPOSES the four modules
above plus the obs/AFD (02-03) and supplementary (02-04) sources into the single
ingestion code path the CLI (backfill) and scheduler (live) both call (D-15). Its
graceful-degradation entry point :func:`ingest_all_models` is re-exported here so
``from weatherquant.ingest import ingest_all_models`` resolves at the package level.
"""

from __future__ import annotations

from weatherquant.ingest.orchestrator import (
    ingest_afd,
    ingest_all_models,
    ingest_cycle,
    ingest_obs,
    ingest_range,
)

__all__ = [
    "ingest_all_models",
    "ingest_cycle",
    "ingest_range",
    "ingest_obs",
    "ingest_afd",
]
