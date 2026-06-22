"""ING-02: nearest-grid-point snap to the Kalshi station (Lambert + lat/lon).

GREEN by 02-02's ``weatherquant.ingest.grib.snap_to_station``. The snap must work on BOTH
the 2-D HRRR Lambert grid and the regular GFS/GEFS lat/lon grids (RESEARCH Pitfall 2) and
return ``grid_distance_m``.
"""

from __future__ import annotations

import pytest

from weatherquant.ingest.grib import decode_t2m, snap_city, snap_to_station

# NYC Kalshi station (KNYC) coords from the registry.
_NYC_LAT, _NYC_LON = 40.779, -73.969


def test_snap_returns_station_cell_and_distance(grib_fixture):
    field = decode_t2m(grib_fixture("hrrr"))
    value_k, distance_m = snap_to_station(field, lat=_NYC_LAT, lon=_NYC_LON)
    assert value_k > 0
    assert distance_m >= 0
    # A ~3 km HRRR grid snaps within a few km of the station.
    assert distance_m < 5_000.0


@pytest.mark.parametrize("name", ["hrrr", "gfs", "gefs"])
def test_snap_works_on_both_grid_types(grib_fixture, name):
    # Lambert (hrrr, 2-D coords) AND regular lat/lon (gfs/gefs, 1-D -> 2-D) both snap.
    field = decode_t2m(grib_fixture(name))
    value_k, distance_m = snap_to_station(field, lat=_NYC_LAT, lon=_NYC_LON)
    assert 200.0 < value_k < 340.0  # physical Kelvin, proves the right cell/unit
    assert distance_m >= 0
    # GFS 0.25° (~28 km) / GEFS 0.5° (~56 km) are coarser; HRRR ~3 km is tight.
    bound = 5_000.0 if name == "hrrr" else 80_000.0
    assert distance_m < bound


def test_snap_normalizes_longitude_for_0_360_grids(grib_fixture):
    # GFS longitude is stored 0..360; the station lon is negative. The snap must normalize
    # so it does NOT land on the antipodal cell (Pitfall 2). A correct snap is close.
    field = decode_t2m(grib_fixture("gfs"))
    _value_k, distance_m = snap_to_station(field, lat=_NYC_LAT, lon=_NYC_LON)
    assert distance_m < 80_000.0  # not the wrong hemisphere


def test_snap_city_resolves_registry_station(grib_fixture):
    field = decode_t2m(grib_fixture("hrrr"))
    temp_k, lat, lon, distance_m = snap_city(field, "NYC")
    assert (lat, lon) == (_NYC_LAT, _NYC_LON)
    assert 200.0 < temp_k < 340.0
    assert distance_m < 5_000.0


def test_snap_distance_bound_raises_on_far_station(grib_fixture):
    # An obviously-out-of-domain station (mid-Pacific) exceeds the HRRR CONUS bound.
    field = decode_t2m(grib_fixture("hrrr"))
    with pytest.raises(ValueError):
        snap_to_station(field, lat=0.0, lon=-160.0, max_distance_m=5_000.0)


async def test_orchestrator_wires_snap_distance_bound(monkeypatch):
    """1.1: the ingest path passes max_distance_m so a far snap raises SanityError (never stored).

    The unit test above proves the guard; this proves the orchestrator actually wires it in —
    on the OLD code (snap_city called without a bound) this would have stored a garbage row.
    """
    from datetime import datetime, timezone

    import numpy as np

    from weatherquant.ingest import orchestrator
    from weatherquant.ingest.errors import SanityError
    from weatherquant.ingest.grib import T2MField

    # A field whose grid sits near (0, 0) — thousands of km from the NYC station.
    far_field = T2MField(
        values=np.array([[295.0, 295.0], [295.0, 295.0]]),
        lat2d=np.array([[0.0, 0.0], [1.0, 1.0]]),
        lon2d=np.array([[0.0, 1.0], [0.0, 1.0]]),
        units="K",
    )
    monkeypatch.setattr(orchestrator.grib, "fetch_t2m", lambda *a, **k: far_field)
    cycle = datetime(2026, 6, 12, 0, tzinfo=timezone.utc)
    # lead != 0 keeps the lead-0 ASOS probe off this path; the snap-distance guard is what fires.
    with pytest.raises(SanityError):
        await orchestrator.ingest_cycle(object(), "hrrr", "NYC", cycle, mode="live", lead=5)
