"""Ingestion package — the shared ingestion spine (Phase 2).

Holds the core spine modules (``available_at`` D-09, ``idempotency`` D-10, ``writer`` D-11,
``grib`` D-02) plus the ``orchestrator`` that composes them with the obs/AFD and supplementary
sources into the one path the CLI and scheduler both call (D-15). The orchestrator's
graceful-degradation entry points are re-exported here for package-level imports.
"""

from __future__ import annotations

from weatherquant.ingest.orchestrator import (
    ingest_afd,
    ingest_cycle,
    ingest_obs,
    ingest_range,
)

__all__ = [
    "ingest_afd",
    "ingest_cycle",
    "ingest_obs",
    "ingest_range",
]
