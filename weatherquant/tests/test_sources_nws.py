"""RED stub — ING-04: NWS gridpoint forecast bucketed into the settlement window.

This is the Wave-0 RED predecessor for ING-04 (Nyquist gap closure); 02-04 turns it GREEN
via ``weatherquant.ingest.sources.nws.fetch_nws_forecast``. The client must set a
descriptive ``User-Agent`` (api.weather.gov 403s without one) and label rows ``nws``.
RED at import until 02-04.
"""

from __future__ import annotations


def test_fetch_nws_forecast_symbol_exists():
    # RED: weatherquant.ingest.sources.nws lands in 02-04 (ImportError until then).
    from weatherquant.ingest.sources.nws import fetch_nws_forecast

    assert callable(fetch_nws_forecast)
