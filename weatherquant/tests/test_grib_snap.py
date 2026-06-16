"""RED stub — ING-02: nearest-grid-point snap to the Kalshi station (Lambert + lat/lon).

Turned GREEN by 02-03's ``weatherquant.ingest.grib.snap_to_station``. The snap must work
on BOTH the 2-D HRRR Lambert grid and the 1-D GFS/GEFS lat/lon grids (RESEARCH Pitfall 2)
and log ``grid_distance_m``. RED at import until 02-03.
"""

from __future__ import annotations


def test_snap_returns_station_cell_and_distance(grib_fixture):
    # RED: weatherquant.ingest.grib lands in 02-03 (ImportError until then).
    from weatherquant.ingest.grib import decode_t2m, snap_to_station

    field = decode_t2m(grib_fixture("hrrr"))
    # NYC Kalshi station coords come from the registry; snap returns (value_k, distance_m).
    value_k, distance_m = snap_to_station(field, lat=40.7790, lon=-73.9692)
    assert value_k > 0
    assert distance_m >= 0
