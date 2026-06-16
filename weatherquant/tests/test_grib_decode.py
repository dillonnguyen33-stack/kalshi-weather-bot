"""ING-01/02: decode a vendored GRIB2 subset to t2m in Kelvin (D-01/D-02/D-03).

GREEN by 02-02's ``weatherquant.ingest.grib``. The fixtures decode offline (vendored in
02-01); ``decode_t2m`` is the edge function that asserts units=="K" and returns a plain
``np.ndarray`` payload (no xarray crosses the boundary — D-02).
"""

from __future__ import annotations

import numpy as np
import pytest

from weatherquant.ingest.grib import T2MField, decode_t2m


@pytest.mark.parametrize("name", ["hrrr", "gfs", "gefs"])
def test_decode_vendored_grib_returns_kelvin_ndarray(grib_fixture, name):
    path = grib_fixture(name)
    field = decode_t2m(path)
    # Decoded 2-m temperature must be in Kelvin (D-03) and a plain ndarray (D-02).
    assert isinstance(field, T2MField)
    assert field.units == "K"
    assert isinstance(field.values, np.ndarray)
    assert field.values.ndim == 2
    # Sanity: physical 2-m temperatures live ~220..330 K, never ~50..110 (those are °F).
    finite = field.values[np.isfinite(field.values)]
    assert finite.min() > 200.0
    assert finite.max() < 340.0


def test_coords_broadcast_to_2d_for_both_grid_types(grib_fixture):
    # HRRR is Lambert (2-D coords); GFS/GEFS are regular lat/lon (1-D -> meshed to 2-D).
    for name in ("hrrr", "gfs", "gefs"):
        field = decode_t2m(grib_fixture(name))
        assert field.lat2d.shape == field.values.shape
        assert field.lon2d.shape == field.values.shape
