"""Wethr.net deterministic forecast outputs (ING-06) — optional key, graceful skip.

Pulls a deterministic model's forecast high from Wethr.net's bearer-authenticated REST API
(``https://wethr.net/api/v2/forecasts.php``), maps the station by the Phase-1 registry
``cli_station`` (NOT v3's stale per-city station map ``NY``/``DAL``/``KDFW``), retries ONCE on
HTTP 429 (Pitfall 7), converts the returned °F high to Kelvin, and stores it under the
provider-namespaced label ``wethr:<model>`` (e.g. ``wethr:hrrr``, ``wethr:nbm``).

GRACEFUL DEGRADATION (D-11). ``Settings.wethr_api_key`` is OPTIONAL (nullable, from 02-01).
When it is unset this module logs a structured skip and returns ``None`` WITHOUT making any
HTTP call — the system proceeds without Wethr. The tests assert the mock client is never
touched in that path (a real key is never required to run the suite).

PROVIDER NAMESPACING (D-12). ``wethr:hrrr`` is a DISTINCT blend input from the NOAA-decoded
``hrrr`` — the two are NEVER deduped/merged, so the same underlying model from two providers
stays two inputs. Rows route through 02-02's :func:`weatherquant.ingest.writer.insert_forecast`
+ :func:`weatherquant.ingest.available_at.available_at` (the single audited write path).

SECRET HYGIENE (T-02-14). The key is read from ``Settings`` (redacted repr, ASVS V14), never
from the environment inline and never logged; the bearer header is sent only to the fixed
wethr.net endpoint (SSRF guard T-02-15).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone

import httpx

from weatherquant.db.engine import get_settings
from weatherquant.ingest.available_at import available_at
from weatherquant.ingest.sources._client import get_client, request_with_retry
from weatherquant.ingest.writer import insert_forecast
from weatherquant.registry import get_city

logger = logging.getLogger(__name__)

# Fixed endpoint (SSRF guard T-02-15) — v3 WETHR_FORECAST_BASE (kalshi_weather_bot_v3.py L528).
WETHR_FORECAST_BASE = "https://wethr.net/api/v2/forecasts.php"

# Deterministic Wethr models in scope (v3 WETHR_MODELS L531). Each stores under wethr:<model>.
WETHR_MODELS = ["hrrr", "nbm", "rap", "nam4km", "gfs", "ecmwf-ifs"]

_KELVIN_OFFSET = 273.15


def fahrenheit_to_kelvin(temp_f: float) -> float:
    """The ONE °F->K conversion on the Wethr path (Pitfall 3 / D-07).

    Wethr returns highs in °F; forecasts are Kelvin-only. Centralizing the conversion keeps
    the units boundary auditable and prevents an accidental °F store into ``temp_kelvin``.
    """
    return (temp_f - 32.0) * 5.0 / 9.0 + _KELVIN_OFFSET


def model_label(model: str) -> str:
    """Provider-namespaced Wethr label (D-12): ``wethr:<model>`` — never deduped with NOAA."""
    return f"wethr:{model.lower()}"


def _extract_high_f(rows: object, target_date: date) -> float | None:
    """Extract the °F high for ``target_date`` from the Wethr ``forecasts.php`` payload (T-02-12).

    The payload is a list of ``{valid_time, temperature_f}`` rows (v3 shape). Rows whose
    ``valid_time`` date matches ``target_date`` contribute their ``temperature_f``; the MAX
    is the daily high. Malformed/None values are skipped (never store garbage). Returns
    ``None`` when no row matches.
    """
    if not isinstance(rows, list):
        return None
    ds = target_date.isoformat()
    highs: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("valid_time", ""))[:10] != ds:
            continue
        temp_f = row.get("temperature_f")
        if temp_f is None:
            continue
        try:
            highs.append(float(temp_f))
        except (TypeError, ValueError):
            continue
    return max(highs) if highs else None


async def fetch_wethr_forecast(
    city: str,
    model: str,
    target_date: date,
    *,
    client: httpx.AsyncClient | None = None,
) -> float | None:
    """Fetch the Wethr forecast high (Kelvin) for ``city``/``model``/``target_date`` (ING-06).

    Reads ``wethr_api_key`` from :func:`get_settings`; if unset, logs a structured skip and
    returns ``None`` WITHOUT any HTTP call (graceful degrade, D-11). Otherwise GETs
    :data:`WETHR_FORECAST_BASE` with ``location_name`` = the registry ``cli_station`` (NOT v3's
    stale per-city station codes), ``model``, ``run=latest``, and ``Authorization: Bearer``;
    a 429 triggers exactly one backoff retry (Pitfall 7). The returned °F high is converted to
    Kelvin. Pure fetch+parse so the unit test injects a ``MockTransport`` client and runs
    offline; the caller persists via :func:`store_wethr_forecast`.

    Args:
        city: Kalshi city code (resolved via :func:`get_city` to its ``cli_station``).
        model: the Wethr model name (stored namespaced as ``wethr:<model>``).
        target_date: the LST settlement (civil) date to forecast the high for.
        client: optional injected ``httpx.AsyncClient`` (the unit test passes a mock).
    """
    api_key = get_settings().wethr_api_key
    if not api_key:
        # Graceful skip (D-11): the key is unset — no HTTP call, log a structured skip.
        logger.warning(
            "Wethr forecast skipped for city=%s model=%s: wethr_api_key unset (degrade, D-11)",
            city,
            model,
        )
        return None

    station = get_city(city).cli_station  # re-map by registry cli_station, NOT v3's station map
    owns_client = client is None
    client = client or get_client(headers={"Authorization": f"Bearer {api_key}"})
    try:
        resp = await request_with_retry(
            client,
            "GET",
            WETHR_FORECAST_BASE,
            params={"location_name": station, "model": model, "run": "latest"},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        resp.raise_for_status()
        rows = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "Wethr fetch failed for city=%s model=%s (%s) — degrading (D-11)",
            city,
            model,
            exc,
        )
        return None
    finally:
        if owns_client:
            await client.aclose()

    high_f = _extract_high_f(rows, target_date)
    return fahrenheit_to_kelvin(high_f) if high_f is not None else None


def store_wethr_forecast(
    bind: object,
    city: str,
    model: str,
    target_date: date,
    temp_kelvin: float,
    *,
    cycle: datetime | None = None,
) -> int:
    """Persist one Wethr forecast row via the SINGLE audited writer path (D-10/D-11).

    Routes through :func:`weatherquant.ingest.writer.insert_forecast` under the provider-
    namespaced label :func:`model_label` (``wethr:<model>``, D-12), ``member=0``, ``lead=0``,
    Kelvin payload, and live ``available_at``. The station snap fields are the registry
    station's own lat/lon with ``grid_distance_m=0.0`` (Wethr returns a station forecast).

    Returns:
        ``1`` if a row was inserted, ``0`` if an identical row already existed (skip).
    """
    station = get_city(city)
    cycle = cycle or datetime.now(timezone.utc)
    return insert_forecast(
        bind,
        city=city,
        target_date=target_date,
        model=model_label(model),
        lead=0,
        member=0,
        temp_kelvin=temp_kelvin,
        cycle=cycle,
        station_lat=station.lat,
        station_lon=station.lon,
        grid_distance_m=0.0,
        # Live fetch -> now(UTC); the live branch ignores the model label (D-09).
        available_at=available_at(cycle, model_label(model), "live"),
    )


__all__ = [
    "WETHR_FORECAST_BASE",
    "WETHR_MODELS",
    "fahrenheit_to_kelvin",
    "model_label",
    "fetch_wethr_forecast",
    "store_wethr_forecast",
]
