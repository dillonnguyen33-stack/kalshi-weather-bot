"""ING-06 (GREEN): Wethr.net bearer-auth source with 429 retry + graceful skip.

02-04 turns the Wave-0 RED stub GREEN via ``weatherquant.ingest.sources.wethr``. Tests mock
httpx (NO network) and the Settings key, proving: (1) with ``wethr_api_key`` UNSET, fetch
returns None and makes NO httpx call (graceful skip — D-11); (2) with a key, every request
carries ``Authorization: Bearer`` and a 429 triggers EXACTLY ONE retry (Pitfall 7); (3) the
station is the registry ``cli_station`` (NOT v3's stale WETHR_STATIONS); (4) the stored label
is provider-namespaced ``wethr:<model>`` (D-12); (5) the °F high converts to a Kelvin value.
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import httpx

from weatherquant.ingest.sources import wethr as wethr_mod
from weatherquant.ingest.sources.wethr import (
    fahrenheit_to_kelvin,
    fetch_wethr_forecast,
    model_label,
)
from weatherquant.registry import get_city

CITY = "CHI"  # cli_station KMDW — proves we map by registry, not v3 WETHR_STATIONS
MODEL = "hrrr"
TARGET = date(2026, 6, 15)


def _set_key(monkeypatch, key: str | None) -> None:
    monkeypatch.setattr(
        wethr_mod, "get_settings", lambda: SimpleNamespace(wethr_api_key=key)
    )


def test_fetch_wethr_forecast_symbol_exists():
    assert callable(fetch_wethr_forecast)


def test_fahrenheit_to_kelvin_centralized():
    assert fahrenheit_to_kelvin(32.0) == 273.15
    assert fahrenheit_to_kelvin(212.0) == 373.15


def test_model_label_provider_namespaced():
    assert model_label("hrrr") == "wethr:hrrr"
    assert model_label("NBM") == "wethr:nbm"


async def test_graceful_skip_when_key_unset_makes_no_http_call(monkeypatch):
    _set_key(monkeypatch, None)
    called: dict = {"hits": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        called["hits"] += 1
        return httpx.Response(200, json=[])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await fetch_wethr_forecast(CITY, MODEL, TARGET, client=client)
    finally:
        await client.aclose()
    assert result is None  # graceful skip
    assert called["hits"] == 0  # NO httpx call when the key is unset (D-11)


async def test_fetch_with_key_sends_bearer_and_cli_station(monkeypatch):
    _set_key(monkeypatch, "secret-token")
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["auth"] = request.headers.get("authorization")
        captured["station"] = request.url.params.get("location_name")
        return httpx.Response(
            200,
            json=[{"valid_time": f"{TARGET.isoformat()} 18:00", "temperature_f": 86.0}],
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await fetch_wethr_forecast(CITY, MODEL, TARGET, client=client)
    finally:
        await client.aclose()
    assert captured["auth"] == "Bearer secret-token"
    # Station is the registry cli_station (KMDW for CHI), not a v3 WETHR_STATIONS code.
    assert captured["station"] == get_city(CITY).cli_station == "KMDW"
    # 86 °F high -> Kelvin (~303 K), in the valid band.
    assert result == fahrenheit_to_kelvin(86.0)
    assert 250.0 <= result <= 330.0


async def test_429_triggers_exactly_one_retry(monkeypatch):
    _set_key(monkeypatch, "secret-token")
    # Avoid a real 10s sleep in the retry path.
    import weatherquant.ingest.sources._client as client_mod

    async def _no_sleep(_seconds):
        return None

    monkeypatch.setattr(client_mod.asyncio, "sleep", _no_sleep)

    hits: dict = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        if hits["n"] == 1:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(
            200,
            json=[{"valid_time": f"{TARGET.isoformat()} 18:00", "temperature_f": 77.0}],
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await fetch_wethr_forecast(CITY, MODEL, TARGET, client=client)
    finally:
        await client.aclose()
    assert hits["n"] == 2  # one 429 + exactly one retry
    assert result == fahrenheit_to_kelvin(77.0)


async def test_utc_valid_time_buckets_into_lst_settlement_day(monkeypatch):
    """1.3: a UTC valid_time past the UTC calendar day still lands in the right LST window.

    CHI (KMDW) is std offset -6, so the 2026-06-15 settlement window is
    [06:00Z 6/15, 06:00Z 6/16). A row at 2026-06-16 04:00Z is the NEXT UTC calendar day but
    STILL inside the 6/15 LST window — the OLD lexical ``[:10]`` match dropped it; the
    settlement_window bucketing keeps it. The 07:00Z row is past end_utc and must be excluded.
    """
    _set_key(monkeypatch, "secret-token")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {"valid_time": "2026-06-16 04:00", "temperature_f": 99.0},  # in 6/15 LST window
                {"valid_time": "2026-06-16 07:00", "temperature_f": 50.0},  # past end → excluded
            ],
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await fetch_wethr_forecast(CITY, MODEL, TARGET, client=client)
    finally:
        await client.aclose()
    # The in-window high is the 99 °F row; the out-of-window 50 °F row is excluded.
    assert result == fahrenheit_to_kelvin(99.0)


def test_store_uses_wethr_label_and_kelvin(monkeypatch):
    captured: dict = {}

    def fake_insert(bind, **kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr(wethr_mod, "insert_forecast", fake_insert)
    rc = wethr_mod.store_wethr_forecast(object(), CITY, MODEL, TARGET, fahrenheit_to_kelvin(80.0))
    assert rc == 1
    assert captured["model"] == "wethr:hrrr"
    assert captured["member"] == 0
    assert captured["lead"] == 0
    assert 250.0 <= captured["temp_kelvin"] <= 330.0
    assert captured["station_lat"] == get_city(CITY).lat
    assert captured["available_at"].tzinfo is not None
