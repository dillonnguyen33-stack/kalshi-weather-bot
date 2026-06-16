"""NOAA GRIB core (ING-01/02) — Herbie byte-range fetch + cfgrib decode + nearest-point snap.

This is the single highest-risk module in Phase 2 (the STATE.md blocker) and the I/O edge
where untrusted NOAA GRIB bytes become a trusted ``np.ndarray``. Four responsibilities:

* :func:`fetch_t2m` — use Herbie to discover the S3 object, log the resolved key + byte
  range + selected ``.idx`` record, subset ``:TMP:2 m`` (~1 MB, NEVER the ~700 MB full
  file), decode with cfgrib, and return a :class:`T2MField` (plain ``np.ndarray`` payload).
* :func:`decode_t2m` — decode an already-downloaded GRIB2 file (the offline test path;
  also the shared decode used by :func:`fetch_t2m`). Asserts decoded units == ``"K"``.
* :func:`snap_to_station` — nearest-grid-point snap (haversine ``argmin``) on the 2-D
  latitude/longitude arrays, working for BOTH Lambert (HRRR/NBM, 2-D coords) and regular
  lat/lon (GFS/GEFS, 1-D coords broadcast to 2-D) grids. Returns ``(temp_kelvin,
  distance_m)`` and logs ``grid_distance_m``.
* :func:`lead0_sanity_check` — the lead-0 acceptance probe: the analysis-time forecast must
  match contemporaneous ASOS within 3 °C (4 °C for DEN, per RESEARCH Pitfall 4); a breach
  raises a loud error.

THE BOUNDARY (D-02). xarray / cfgrib NEVER leave this module — the decoded field crosses
out as a plain :class:`T2MField` whose ``.values`` is a NumPy array. No xarray ``Dataset``
appears in any public return annotation. THE UNIT GUARD (D-03 / Pitfall 3): the decode
asserts the cfgrib ``units`` attribute is exactly ``"K"`` before the field is trusted;
forecasts are Kelvin-only and °F never enters this path.

GEFS member mapping (D-05): ``c00`` -> ``member=0``; ``p01..p30`` -> ``member=1..30``;
deterministic models (HRRR/GFS/NBM) write ``member=0``. The sync Herbie+cfgrib decode is
CPU-bound; callers on the async loop run :func:`fetch_t2m` in a thread executor (D-14), but
the function itself stays sync and is unit-tested against the offline vendored fixtures.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from weatherquant.ingest.errors import SanityError, UnitError
from weatherquant.registry import get_city

# 2-D float field / coordinate array at the GRIB I/O edge. GRIB temps decode as float32
# and cfgrib coords as float64, so the element type is left as any floating kind.
FloatArray = npt.NDArray[np.floating[Any]]

logger = logging.getLogger(__name__)

_EARTH_RADIUS_M = 6_371_000.0  # mean Earth radius (haversine), meters.

# Lead-0 sanity tolerance (D-04 / RESEARCH Pitfall 4). Default 3 °C; DEN (Denver, 1656 m,
# complex terrain) is explicitly authorized a wider 4 °C band, logged when applied.
_LEAD0_TOLERANCE_C = 3.0
_LEAD0_TOLERANCE_C_DEN = 4.0

# GEFS Herbie member labels -> the forecasts.member smallint axis (D-05).
#   c00 (control) -> 0 ; p01..p30 -> 1..30 ; deterministic models -> 0.
_GEFS_DETERMINISTIC_MEMBER = 0


@dataclass(frozen=True)
class T2MField:
    """Decoded 2-m temperature field at the I/O edge — NumPy only, no xarray (D-02).

    Attributes:
        values: the 2-D temperature field in Kelvin (``np.ndarray``, shape ``(ny, nx)``).
        lat2d: 2-D latitude array aligned with ``values`` (degrees north).
        lon2d: 2-D longitude array aligned with ``values`` (degrees east, may be 0..360).
        units: the decoded unit string — asserted ``"K"`` at decode time (D-03).
    """

    values: FloatArray
    lat2d: FloatArray
    lon2d: FloatArray
    units: str


def member_to_axis(member: str) -> int:
    """Map a Herbie GEFS member label to the ``forecasts.member`` axis (D-05).

    ``"c00"`` -> 0 (control); ``"p01".."p30"`` -> 1..30. Any other label (deterministic
    models, ``avg``/``spr``, or an explicit control) resolves to 0. Never a silent wrong
    index — an out-of-range perturbation label raises ``ValueError``.
    """
    member = member.lower()
    if member in ("c00", "", "avg", "mean", "spr", "control"):
        return _GEFS_DETERMINISTIC_MEMBER
    if member.startswith("p"):
        idx = int(member[1:])
        if not 1 <= idx <= 30:
            raise ValueError(f"GEFS perturbation member out of range 1..30: {member!r}")
        return idx
    raise ValueError(f"unrecognized GEFS member label: {member!r}")


def _to_2d(lat: FloatArray, lon: FloatArray, shape: tuple[int, int]) -> tuple[FloatArray, FloatArray]:
    """Broadcast coordinate arrays to 2-D aligned with the field (Pitfall 2).

    HRRR/NBM are Lambert: cfgrib already exposes 2-D ``latitude``/``longitude``. GFS/GEFS
    are regular lat/lon: cfgrib exposes 1-D coords that must be meshed to 2-D so the same
    haversine ``argmin`` works for both grid types.
    """
    if lat.ndim == 2 and lon.ndim == 2:
        return lat, lon
    # 1-D regular grid -> meshgrid to (ny, nx) matching the field shape.
    lon2d, lat2d = np.meshgrid(lon, lat)
    if lat2d.shape != shape:
        raise ValueError(
            f"broadcast coord shape {lat2d.shape} != field shape {shape}"
        )
    return lat2d, lon2d


def _open_t2m_field(path: Path) -> T2MField:
    """Open a GRIB2 file with cfgrib and extract the 2-m temperature as a :class:`T2MField`.

    xarray/cfgrib are confined to THIS private helper (D-02) — the labeled DataArray is
    converted to plain ``np.ndarray`` here and never crosses out; callers receive only the
    NumPy-backed :class:`T2MField`. The ``indexpath=""`` keeps cfgrib from writing a sidecar
    ``.idx`` next to the (possibly read-only) vendored fixture. The Kelvin units guard
    (Pitfall 3) runs here so a non-Kelvin field is rejected before any NumPy leaves.
    """
    import xarray as xr  # imported lazily so the boundary stays inside this module (D-02).

    ds = xr.open_dataset(
        str(path),
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    )
    if "t2m" not in ds:
        raise ValueError(f"no t2m (TMP:2 m) variable in GRIB file: {path}")
    da = ds["t2m"]
    units = str(da.attrs.get("units", ""))
    if units != "K":
        # Pitfall 3: a non-Kelvin forecast field is a hard error — °F/°C must never enter
        # the Kelvin-only forecast path (D-07). A CorrectnessError (UnitError) so the
        # orchestrator fails LOUD instead of degrading it to a silent skip (WR-05). Still a
        # ValueError ("units must be 'K'") so the existing units-boundary test holds.
        raise UnitError(
            f"decoded TMP:2 m units must be 'K' (Kelvin-only forecast path, D-03/D-07); "
            f"got units={units!r}"
        )
    values = np.asarray(da.values)
    if values.ndim != 2:
        raise ValueError(f"expected a 2-D t2m field, got ndim={values.ndim}")
    lat = np.asarray(da["latitude"].values)
    lon = np.asarray(da["longitude"].values)
    lat2d, lon2d = _to_2d(lat, lon, values.shape)
    return T2MField(values=values, lat2d=lat2d, lon2d=lon2d, units=units)


def decode_t2m(path: str | Path) -> T2MField:
    """Decode a GRIB2 file to a Kelvin :class:`T2MField` (D-02 / D-03).

    Thin public wrapper over :func:`_open_t2m_field`: asserts the cfgrib ``units`` attribute
    is exactly ``"K"`` (Pitfall 3), converts the labeled DataArray to a plain ``np.ndarray``
    at the edge, and broadcasts the coordinate arrays to 2-D so the snap works for both
    Lambert and lat/lon grids. No xarray object crosses this public boundary (D-02).

    Raises:
        ValueError: if the file has no ``t2m`` variable or its units are not ``"K"``.
    """
    return _open_t2m_field(Path(path))


def _haversine_m(lat1: float, lon1: float, lat2: FloatArray, lon2: FloatArray) -> FloatArray:
    """Great-circle distance (meters) from one point to every grid cell (vectorized).

    Longitudes are normalized to [-180, 180) before the difference so a 0..360 GFS grid and
    a negative station longitude compare correctly (Pitfall 2 — wrong-cell snaps otherwise).
    """
    lat1r = math.radians(lat1)
    lon1r = math.radians(((lon1 + 180.0) % 360.0) - 180.0)
    lat2r = np.radians(lat2)
    lon2r = np.radians(((lon2 + 180.0) % 360.0) - 180.0)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + math.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    dist: FloatArray = 2.0 * _EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
    return dist


def snap_to_station(
    field: T2MField,
    *,
    lat: float,
    lon: float,
    max_distance_m: float | None = None,
) -> tuple[float, float]:
    """Nearest-grid-point snap to ``(lat, lon)`` on the 2-D coords (D-03 / Pitfall 2).

    Computes the haversine distance from the station to EVERY grid cell on the 2-D
    latitude/longitude arrays (works for Lambert HRRR/NBM and regular GFS/GEFS) and returns
    the temperature at the closest cell plus that distance. Logs ``grid_distance_m``.

    Args:
        field: a decoded :class:`T2MField`.
        lat: station latitude (degrees north).
        lon: station longitude (degrees east; negative West is fine — normalized internally).
        max_distance_m: optional sane-bound assertion. A ~3 km HRRR/NBM grid should snap
            within a few km; GFS 0.25° / GEFS 0.5° are coarser. A snap farther than this
            raises (a bad station coord / grid mislabel — Pitfall 2). ``None`` disables.

    Returns:
        ``(temp_kelvin, distance_m)`` — the Kelvin value at the nearest cell and the
        great-circle distance (meters) from the station to that cell.
    """
    dist = _haversine_m(lat, lon, field.lat2d, field.lon2d)
    flat_idx = int(np.argmin(dist))
    iy, ix = np.unravel_index(flat_idx, dist.shape)
    distance_m = float(dist[iy, ix])
    temp_kelvin = float(field.values[iy, ix])
    logger.info(
        "grib snap station=(%.4f,%.4f) -> cell=(%.4f,%.4f) grid_distance_m=%.1f temp_k=%.2f",
        lat,
        lon,
        float(field.lat2d[iy, ix]),
        float(field.lon2d[iy, ix]),
        distance_m,
        temp_kelvin,
    )
    if max_distance_m is not None and distance_m > max_distance_m:
        # Snap-distance breach is a correctness alarm (SanityError): a bad station coord / grid
        # mislabel must fail LOUD, not degrade to a silent skip (WR-05). Still a ValueError.
        raise SanityError(
            f"nearest grid point {distance_m:.0f} m from station exceeds bound "
            f"{max_distance_m:.0f} m — likely a bad station coord or grid mislabel "
            f"(Pitfall 2)"
        )
    return temp_kelvin, distance_m


def snap_city(
    field: T2MField,
    city_code: str,
    *,
    max_distance_m: float | None = None,
) -> tuple[float, float, float, float]:
    """Snap ``field`` to a registry city's Kalshi station (D-03).

    Resolves the station lat/lon via :func:`weatherquant.registry.get_city` (raises
    ``KeyError`` on an unknown code — ASVS V5, never a silent default) and returns
    ``(temp_kelvin, station_lat, station_lon, grid_distance_m)`` for the forecast row.
    """
    city = get_city(city_code)
    temp_kelvin, distance_m = snap_to_station(
        field, lat=city.lat, lon=city.lon, max_distance_m=max_distance_m
    )
    return temp_kelvin, city.lat, city.lon, distance_m


def lead0_sanity_check(
    *,
    forecast_k: float,
    asos_k: float,
    tolerance_c: float | None = None,
    city_code: str | None = None,
) -> None:
    """Assert the lead-0 forecast matches contemporaneous ASOS (D-04 / Pitfall 4).

    A model analysis (lead 0) is constrained by assimilated obs and should sit within a few
    degrees of the station. A breach is almost always a wrong station snap / unit error /
    grid mislabel (often 5-15 °C off; a K-vs-°C swap is ~250 °C off), so it raises LOUDLY
    rather than letting corrupt data into calibration.

    Both temperatures are Kelvin (a Kelvin difference equals a Celsius difference, so the
    tolerance is in °C directly). Compare at the SAME instant (the caller snaps ASOS to the
    analysis hour).

    Args:
        forecast_k: the lead-0 forecast temperature (Kelvin).
        asos_k: the contemporaneous ASOS observation at the analysis hour (Kelvin).
        tolerance_c: explicit tolerance override (°C). If ``None``, defaults to 3 °C, or
            4 °C when ``city_code == "DEN"`` (the authorized wide band, logged).
        city_code: optional city; ``"DEN"`` relaxes the default tolerance to 4 °C.

    Raises:
        SanityError: if ``|forecast_k - asos_k|`` exceeds the tolerance (loud breach). It is a
            ``CorrectnessError`` (so the orchestrator fails loud, WR-05) and a ``ValueError``.
    """
    if tolerance_c is None:
        if city_code == "DEN":
            tolerance_c = _LEAD0_TOLERANCE_C_DEN
            logger.info(
                "lead-0 probe: DEN high-elevation band -> tolerance %.1f C (Pitfall 4)",
                tolerance_c,
            )
        else:
            tolerance_c = _LEAD0_TOLERANCE_C
    delta_c = abs(forecast_k - asos_k)
    if delta_c > tolerance_c:
        # Lead-0 breach is a correctness alarm (SanityError): a wrong snap / unit / grid error
        # must fail LOUD, not degrade to a silent skip (WR-05). Still a ValueError.
        raise SanityError(
            f"lead-0 sanity breach: |forecast {forecast_k:.2f}K - ASOS {asos_k:.2f}K| = "
            f"{delta_c:.2f} C > {tolerance_c:.1f} C tolerance"
            + (f" (city={city_code})" if city_code else "")
            + " — wrong station snap / unit error / grid mislabel (D-04, Pitfall 4)"
        )
    logger.info(
        "lead-0 probe OK: delta=%.2f C <= %.1f C tolerance%s",
        delta_c,
        tolerance_c,
        f" (city={city_code})" if city_code else "",
    )


def fetch_t2m(
    model: str,
    cycle_init: datetime,
    fxx: int,
    member: str = "c00",
) -> T2MField:
    """Fetch + decode the ``:TMP:2 m`` byte-range subset for one model run (ING-01/02).

    Uses Herbie to discover the S3 object, logs the resolved S3 key + byte range + selected
    ``.idx`` record (D-01), subsets ONLY ``:TMP:2 m`` (~1 MB, never the full ~700 MB file),
    and decodes the subset with cfgrib via :func:`decode_t2m`. Returns a Kelvin
    :class:`T2MField` (NumPy at the edge, D-02). This is the sync, CPU-bound path; async
    callers run it in a thread executor (D-14). It is NOT exercised by the offline unit
    tests (which call :func:`decode_t2m` on the vendored fixtures directly) — it is the live
    ingestion entry point 02-05 schedules.

    Args:
        model: ``"hrrr"`` / ``"gfs"`` / ``"gefs"`` / ``"nbm"``.
        cycle_init: the model run init time (UTC).
        fxx: forecast lead hours.
        member: GEFS member label (``"c00"``/``"p01".."p30"``); ignored for deterministic
            models. See :func:`member_to_axis` for the member -> axis mapping (D-05).

    Returns:
        A decoded Kelvin :class:`T2MField` for the requested run.
    """
    from herbie import Herbie  # lazy import: keep Herbie off the offline test path.

    # Herbie's _validate() compares the run date against a tz-NAIVE pandas Timestamp
    # (``pd.Timestamp.utcnow().tz_localize(None)``), so a tz-aware ``cycle_init`` raises
    # "Cannot compare tz-naive and tz-aware timestamps". Herbie treats the run time as UTC
    # implicitly; normalize an aware datetime to a naive UTC instant before handing it over.
    if cycle_init.tzinfo is not None:
        herbie_date = cycle_init.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        herbie_date = cycle_init

    h_kwargs: dict[str, object] = {"model": model, "fxx": fxx}
    if model == "gefs":
        h_kwargs["member"] = member
    herbie = Herbie(herbie_date, **h_kwargs)

    # Log the resolved S3 key + the selected .idx record / byte range (D-01) before fetch.
    try:
        inventory = herbie.inventory(":TMP:2 m")
        logger.info(
            "herbie resolved model=%s cycle=%s fxx=%s member=%s grib_source=%s "
            "idx_records=%d",
            model,
            cycle_init.isoformat(),
            fxx,
            member if model == "gefs" else _GEFS_DETERMINISTIC_MEMBER,
            getattr(herbie, "grib", None),
            len(inventory),
        )
    except Exception:  # noqa: BLE001 - logging probe must never block the fetch.
        logger.warning("herbie .idx inventory probe failed for model=%s; proceeding", model)

    # Download ONLY the :TMP:2 m byte-range subset (~1 MB, never the full ~700 MB file). The
    # ``search`` argument is the GRIB message filter; herbie-data 2026.3.0 dropped the old
    # ``remove_grib`` kwarg, so the subset file is kept by default and decoded below.
    local_path = herbie.download(":TMP:2 m")
    return decode_t2m(local_path)


__all__ = [
    "T2MField",
    "decode_t2m",
    "fetch_t2m",
    "snap_to_station",
    "snap_city",
    "lead0_sanity_check",
    "member_to_axis",
]
