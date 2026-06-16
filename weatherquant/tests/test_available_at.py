"""RED stub — ING-09 / D-09: available_at = cycle_init + latency (backfill) vs now (live).

Turned GREEN by 02-02's ``weatherquant.ingest.available_at``. This is the worst
look-ahead landmine in the phase (Pitfall 5): backfill MUST stamp ``cycle_init +
PUBLISH_LATENCY[model]`` and NEVER ``now()``; live stamps ``now()``. The NBM latency is
the Wave-0 probe value recorded in 02-01. RED at import until 02-02.
"""

from __future__ import annotations

from datetime import datetime, timezone


def test_backfill_uses_cycle_plus_latency_not_now():
    # RED: weatherquant.ingest.available_at lands in 02-02 (ImportError until then).
    from weatherquant.ingest.available_at import PUBLISH_LATENCY, available_at

    cycle = datetime(2026, 6, 13, 12, 0, tzinfo=timezone.utc)
    backfilled = available_at(cycle, "hrrr", mode="backfill")
    # Backfill is deterministic: cycle_init + the model's publish latency, never now().
    assert backfilled == cycle + PUBLISH_LATENCY["hrrr"]
    assert "nbm" in PUBLISH_LATENCY  # the Wave-0 probe value (02-01) feeds this
