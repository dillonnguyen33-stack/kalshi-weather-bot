"""NOAA GRIB core (ING-01/02) — Herbie byte-range fetch + cfgrib decode + nearest-point snap.

The I/O edge where untrusted GRIB bytes become a trusted ``np.ndarray``: fetch/decode the
``:TMP:2 m`` subset, snap to the nearest grid point, and probe lead-0 sanity. xarray/cfgrib
never leave this module (the field crosses out as a plain :class:`T2MField`), the decode
asserts units == ``"K"``, and GEFS members map c00→0 / p01..p30→1..30 (D-02/D-03/D-05; see
docs/DECISIONS.md). The sync decode is CPU-bound; async callers run :func:`fetch_t2m` off-loop
(D-14).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, UTC
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
        values: 2-D Kelvin field, shape ``(ny, nx)``.
        lat2d/lon2d: 2-D coords aligned with ``values`` (lon may be 0..360).
        units: decoded unit string — asserted ``"K"`` at decode time (D-03).
    """

    values: FloatArray
    lat2d: FloatArray
    lon2d: FloatArray
    units: str


def member_to_axis(member: str) -> int:
    """Map a Herbie GEFS member label to the ``forecasts.member`` axis (D-05).

    ``c00``/``avg``/``spr``/``control`` → 0; ``p01..p30`` → 1..30. An out-of-range
    perturbation label raises ``ValueError`` (never a silent wrong index).
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
    """Broadcast coordinate arrays to 2-D aligned with the field so one snap works for both grid types (Pitfall 2).

    HRRR/NBM Lambert coords are already 2-D; GFS/GEFS regular 1-D coords are meshed to 2-D.
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

    Confines xarray/cfgrib to this helper (D-02), converting to ``np.ndarray`` at the edge;
    ``indexpath=""`` avoids a sidecar ``.idx`` next to a read-only fixture, and the Kelvin
    units guard runs here before any NumPy leaves (Pitfall 3).
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
        # Pitfall 3: a non-Kelvin field is a hard error (D-07). A CorrectnessError (UnitError)
        # so the orchestrator fails loud, never a silent skip (WR-05); still a ValueError so the
        # units-boundary test holds.
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
    """Decode a GRIB2 file to a Kelvin :class:`T2MField`, asserting units == ``"K"`` (D-02 / D-03).

    Raises:
        ValueError: if the file has no ``t2m`` variable or its units are not ``"K"``.
    """
    return _open_t2m_field(Path(path))


def _haversine_m(lat1: float, lon1: float, lat2: FloatArray, lon2: FloatArray) -> FloatArray:
    """Great-circle distance (meters) from one point to every grid cell, vectorized.

    Longitudes are normalized to [-180, 180) so a 0..360 grid and a negative station longitude
    compare correctly (Pitfall 2).
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
    """Nearest-grid-point snap to ``(lat, lon)`` via haversine ``argmin`` (D-03 / Pitfall 2).

    Args:
        max_distance_m: optional sane-bound; a snap farther than this raises (bad coord / grid
            mislabel, Pitfall 2). ``None`` disables.

    Returns:
        ``(temp_kelvin, distance_m)`` at the nearest cell.
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
    """Snap ``field`` to a registry city's Kalshi station, returning the forecast-row fields (D-03).

    Resolves the station via :func:`get_city` (raises ``KeyError`` on unknown — ASVS V5) and
    returns ``(temp_kelvin, station_lat, station_lon, grid_distance_m)``.
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
    """Assert the lead-0 forecast matches contemporaneous ASOS, else fail loud (D-04 / Pitfall 4).

    Both temps are Kelvin (so the °C tolerance applies directly) and compared at the same
    instant; a breach is almost always a wrong snap / unit / grid error.

    Args:
        tolerance_c: override (°C). ``None`` → 3 °C, or 4 °C for ``city_code == "DEN"`` (logged).
        city_code: optional; ``"DEN"`` relaxes the default tolerance to 4 °C.

    Raises:
        SanityError: if ``|forecast_k - asos_k|`` exceeds the tolerance. A ``CorrectnessError``
            (orchestrator fails loud, WR-05) and a ``ValueError``.
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

    Discovers the S3 object via Herbie, logs the resolved key + byte range (D-01), subsets only
    ``:TMP:2 m`` (~1 MB, never the ~700 MB full file), and decodes via :func:`decode_t2m`. The
    sync, CPU-bound live path (async callers run it off-loop, D-14); the offline tests call
    :func:`decode_t2m` directly instead.

    Args:
        member: GEFS member label, ignored for deterministic models (see :func:`member_to_axis`).

    Returns:
        A decoded Kelvin :class:`T2MField` for the requested run.
    """
    from herbie import Herbie  # lazy import: keep Herbie off the offline test path.

    # Herbie's _validate() compares against a tz-naive Timestamp, so an aware ``cycle_init``
    # raises "Cannot compare tz-naive and tz-aware"; normalize to a naive UTC instant first.
    if cycle_init.tzinfo is not None:
        herbie_date = cycle_init.astimezone(UTC).replace(tzinfo=None)
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

    # Download ONLY the :TMP:2 m byte-range subset (~1 MB). herbie-data 2026.3.0 dropped the
    # old ``remove_grib`` kwarg, so the subset file is kept by default and decoded below.
    local_path = herbie.download(":TMP:2 m")
    return decode_t2m(local_path)


__all__ = [
    "T2MField",
    "decode_t2m",
    "fetch_t2m",
    "lead0_sanity_check",
    "member_to_axis",
    "snap_city",
    "snap_to_station",
]
