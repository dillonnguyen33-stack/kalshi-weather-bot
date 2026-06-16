"""RED stub — ING-08 / D-11: a missing model cycle is logged + skipped, others ingest.

Turned GREEN by 02-05's scheduler/CLI ingestion path. A single failed/missing model
cycle must degrade gracefully (logged fallback) without aborting the other models'
ingestion. RED at import until the ingest entrypoint exists.
"""

from __future__ import annotations


def test_missing_cycle_does_not_abort_other_models():
    # RED: weatherquant.ingest entrypoint lands in 02-05 (ImportError until then).
    from weatherquant.ingest import ingest_all_models

    assert callable(ingest_all_models)
