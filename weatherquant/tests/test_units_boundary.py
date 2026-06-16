"""RED stub — ING-01: decoded forecast units are Kelvin; degF never enters forecasts.

Turned GREEN by 02-03's ``weatherquant.ingest.grib``. The forecast path is Kelvin-only
(D-07): the degF conversion is confined to the observations path (obs stay degF). This
guard asserts the GRIB decode edge refuses / never emits a non-Kelvin forecast value.
RED at import until 02-03.
"""

from __future__ import annotations


def test_forecast_units_are_kelvin(grib_fixture):
    # RED: weatherquant.ingest.grib lands in 02-03 (ImportError until then).
    from weatherquant.ingest.grib import decode_t2m

    field = decode_t2m(grib_fixture("gfs"))
    assert field.units == "K"  # forecasts are Kelvin-only; degF never enters (D-07)
