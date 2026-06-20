"""Wethr.net deterministic forecast outputs (ING-06) — optional key, graceful skip.

Pulls a deterministic model's high from Wethr.net's bearer API, keyed by the registry
``cli_station`` (not v3's stale map), retries once on 429 (Pitfall 7), converts °F → Kelvin,
and stores under the namespaced label ``wethr:<model>`` (distinct from NOAA ``hrrr``, D-12).
An unset ``wethr_api_key`` logs a skip and returns ``None`` with no HTTP call (D-11). The key
is read from ``Settings`` (redacted, never logged) and sent only to the fixed endpoint
(T-02-14/T-02-15; see docs/DECISIONS.md).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, UTC

import httpx

from weatherquant.db.engine import get_settings
from weatherquant.ingest.available_at import available_at
from weatherquant.ingest.sources._client import get_client, request_with_retry
from weatherquant.ingest.writer import Bind, insert_forecast
from weatherquant.registry import get_city

logger = logging.getLogger(__name__)

# Fixed endpoint (SSRF guard T-02-15) — v3 WETHR_FORECAST_BASE (kalshi_weather_bot_v3.py L528).
WETHR_FORECAST_BASE = "https://wethr.net/api/v2/forecasts.php"

# Deterministic Wethr models in scope (v3 WETHR_MODELS L531). Each stores under wethr:<model>.
WETHR_MODELS = ["hrrr", "nbm", "rap", "nam4km", "gfs", "ecmwf-ifs"]

_KELVIN_OFFSET = 273.15


def fahrenheit_to_kelvin(temp_f: float) -> float:
    """The ONE °F->K conversion on the Wethr path, keeping the units boundary auditable (D-07)."""
    return (temp_f - 32.0) * 5.0 / 9.0 + _KELVIN_OFFSET


def model_label(model: str) -> str:
    """Provider-namespaced Wethr label (D-12): ``wethr:<model>`` — never deduped with NOAA."""
    return f"wethr:{model.lower()}"


def _extract_high_f(rows: object, target_date: date) -> float | None:
    """Extract the °F high for ``target_date`` from the Wethr payload, else ``None`` (T-02-12).

    Rows are ``{valid_time, temperature_f}``; the max over date-matching rows is the high,
    malformed values skipped.
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

    Unset ``wethr_api_key`` → skip + ``None`` with no HTTP call (degrade, D-11). Otherwise GETs
    keyed by the registry ``cli_station`` with one 429 retry (Pitfall 7), converting °F → Kelvin.
    Pure fetch+parse so the unit test injects a ``MockTransport`` client offline.

    Args:
        model: the Wethr model name (stored namespaced as ``wethr:<model>``).
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
    bind: Bind,
    city: str,
    model: str,
    target_date: date,
    temp_kelvin: float,
    *,
    cycle: datetime | None = None,
    mode: str = "live",
) -> int:
    """Persist one Wethr forecast row via the SINGLE audited writer path (D-10/D-11).

    Stores under :func:`model_label` (``wethr:<model>``, D-12), ``member=0``, ``lead=0``, Kelvin
    payload, station snap = registry lat/lon with ``grid_distance_m=0.0``, and ``available_at``
    threaded with ``mode`` (WR-01, seam stays single — D-15; orchestrator refuses backfill, WR-02).

    Returns:
        ``1`` if a row was inserted, ``0`` if an identical row already existed (skip).
    """
    station = get_city(city)
    cycle = cycle or datetime.now(UTC)
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
        # available_at honors the threaded mode (WR-01); the live branch ignores the label.
        available_at=available_at(cycle, model_label(model), mode),  # type: ignore[arg-type]
    )


__all__ = [
    "WETHR_FORECAST_BASE",
    "WETHR_MODELS",
    "fahrenheit_to_kelvin",
    "fetch_wethr_forecast",
    "model_label",
    "store_wethr_forecast",
]
