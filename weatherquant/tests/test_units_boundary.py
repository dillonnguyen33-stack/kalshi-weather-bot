"""ING-01: decoded forecast units are Kelvin; degF never enters forecasts.

GREEN by 02-02's ``weatherquant.ingest.grib``. The forecast path is Kelvin-only (D-07):
the degF conversion is confined to the observations path. The GRIB decode edge refuses any
non-Kelvin field (Pitfall 3).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from weatherquant.ingest.grib import decode_t2m


def test_forecast_units_are_kelvin(grib_fixture):
    field = decode_t2m(grib_fixture("gfs"))
    assert field.units == "K"  # forecasts are Kelvin-only; degF never enters (D-07)


def test_decode_rejects_non_kelvin_field(monkeypatch):
    # The units guard (Pitfall 3) must raise on any non-Kelvin field. Monkeypatch
    # xarray.open_dataset (the only cfgrib boundary, confined to grib.py) to return a
    # Fahrenheit-labeled field; decode_t2m must reject it rather than let °F into the
    # Kelvin-only forecast path (D-07).
    fahrenheit = xr.Dataset(
        {
            "t2m": xr.DataArray(
                np.full((3, 4), 75.0),
                dims=("latitude", "longitude"),
                coords={"latitude": np.arange(3.0), "longitude": np.arange(4.0)},
                attrs={"units": "F"},
            )
        }
    )
    monkeypatch.setattr(xr, "open_dataset", lambda *a, **k: fahrenheit)
    with pytest.raises(ValueError, match="units must be 'K'"):
        from weatherquant.ingest.grib import decode_t2m as _decode

        _decode("ignored.grib2")


def test_all_vendored_fixtures_are_kelvin(grib_fixture):
    for name in ("hrrr", "gfs", "gefs"):
        field = decode_t2m(grib_fixture(name))
        assert field.units == "K"
        finite = field.values[np.isfinite(field.values)]
        # Kelvin range, never °F (~50..110): proves no unit leak into the forecast path.
        assert finite.min() > 200.0
