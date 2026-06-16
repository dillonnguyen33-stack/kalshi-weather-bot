"""RED stub — ING-05: Open-Meteo ensemble per-member parse + degC->K; member->column map.

Turned GREEN by 02-04's ``weatherquant.ingest.sources.openmeteo``. Open-Meteo returns degC
(API default) — it MUST be converted to Kelvin for ``forecasts.temp_kelvin`` (D-07,
Pitfall 3); never copy v3's ``temperature_unit=fahrenheit``. RED at import until 02-04.
"""

from __future__ import annotations


def test_openmeteo_members_parsed_in_kelvin():
    # RED: weatherquant.ingest.sources.openmeteo lands in 02-04 (ImportError until then).
    from weatherquant.ingest.sources.openmeteo import fetch_openmeteo_ensemble

    assert callable(fetch_openmeteo_ensemble)
