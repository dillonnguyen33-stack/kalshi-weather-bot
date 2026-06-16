"""RED stub — ING-03: daily-high = max over the LST settlement_window; obs_count logged.

Turned GREEN by 02-04's ``weatherquant.ingest.obs``. The daily high MUST be bucketed
through ``weatherquant.time.settlement_window`` (half-open [start, end), D-16) — never a
hand-rolled UTC day. RED at import until 02-04.
"""

from __future__ import annotations


def test_daily_high_is_max_over_settlement_window():
    # RED: weatherquant.ingest.obs lands in 02-04 (ImportError until then).
    from weatherquant.ingest.obs import daily_high_from_obs

    result = daily_high_from_obs(city="NYC", target_date=None, readings=[])
    assert hasattr(result, "daily_high_f")
    assert hasattr(result, "obs_count")
