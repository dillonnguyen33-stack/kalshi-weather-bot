"""RED stub — ING-06: Wethr.net bearer-auth source with 429 retry + graceful skip.

This is the Wave-0 RED predecessor for ING-06 (Nyquist gap closure); 02-04 turns it GREEN
via ``weatherquant.ingest.sources.wethr.fetch_wethr_forecast``. The client uses
``Authorization: Bearer {WETHR_API_KEY}``, retries once on 429, skips gracefully when the
key is absent (D-11), and labels rows ``wethr:<model>``. RED at import until 02-04.
"""

from __future__ import annotations


def test_fetch_wethr_forecast_symbol_exists():
    # RED: weatherquant.ingest.sources.wethr lands in 02-04 (ImportError until then).
    from weatherquant.ingest.sources.wethr import fetch_wethr_forecast

    assert callable(fetch_wethr_forecast)
