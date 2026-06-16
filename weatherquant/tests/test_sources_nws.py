"""ING-04 (GREEN): NWS gridpoint forecast bucketed into the LST settlement window.

02-04 turns the Wave-0 RED stub GREEN via ``weatherquant.ingest.sources.nws``. These tests
mock httpx (``httpx.MockTransport`` — NO network) and prove: (1) the two-hop /points ->
forecastGridData flow parses ``properties.temperature`` and buckets its ISO-interval values
into ``settlement_window`` (a hotter value on the wrong LST day does NOT win); (2) the
explicit ``uom`` is converted to Kelvin (a ``degC`` payload lands in the ~280-320 K band,
never ~50-110 °F); (3) every request carries a User-Agent (api.weather.gov 403s without one);
(4) the stored forecast row uses model label ``nws`` and ``member=0``.
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import httpx

from weatherquant.ingest.sources import nws as nws_mod
from weatherquant.ingest.sources.nws import (
    MODEL,
    fetch_nws_forecast,
    window_max_kelvin,
)
from weatherquant.registry import get_city
from weatherquant.time import settlement_window

CITY = "NYC"
TARGET = date(2026, 6, 15)


def _grid_payload(values: list[dict]) -> dict:
    return {"properties": {"temperature": {"uom": "wmoUnit:degC", "values": values}}}


def _build_mock_client(captured: dict) -> httpx.AsyncClient:
    """An httpx client whose MockTransport serves the /points then /gridpoints hops offline."""
    win = settlement_window(get_city(CITY), TARGET)
    grid_url = "https://api.weather.gov/gridpoints/OKX/33,35"
    # Three in-window degC hours (peak 30 C -> 303.15 K) plus a hotter value the day BEFORE
    # the window (must NOT win — proves correct bucketing, not a flat window).
    hot_before = win.start_utc - timedelta(hours=3)
    in_a = win.start_utc + timedelta(hours=14)
    in_b = win.start_utc + timedelta(hours=18)
    in_c = win.start_utc + timedelta(hours=20)
    grid = _grid_payload(
        [
            {"validTime": f"{hot_before.isoformat()}/PT1H", "value": 45.0},  # hotter, wrong day
            {"validTime": f"{in_a.isoformat()}/PT1H", "value": 22.0},
            {"validTime": f"{in_b.isoformat()}/PT1H", "value": 30.0},  # the in-window peak
            {"validTime": f"{in_c.isoformat()}/PT1H", "value": 27.0},
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        captured.setdefault("user_agents", []).append(request.headers.get("user-agent"))
        if "/points/" in request.url.path:
            return httpx.Response(
                200, json={"properties": {"forecastGridData": grid_url}}
            )
        if "/gridpoints/" in request.url.path:
            return httpx.Response(200, content=json.dumps(grid))
        return httpx.Response(404)

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={"User-Agent": "weatherquant/0.1 (test)"},
    )


def test_fetch_nws_forecast_symbol_exists():
    assert callable(fetch_nws_forecast)


def test_window_max_kelvin_buckets_and_converts_degc():
    win = settlement_window(get_city(CITY), TARGET)
    hot_before = win.start_utc - timedelta(hours=2)
    in_peak = win.start_utc + timedelta(hours=16)
    temperature = {
        "uom": "wmoUnit:degC",
        "values": [
            {"validTime": f"{hot_before.isoformat()}/PT1H", "value": 50.0},  # wrong day
            {"validTime": f"{in_peak.isoformat()}/PT1H", "value": 25.0},  # in-window peak
        ],
    }
    result = window_max_kelvin(temperature, win)
    assert result is not None
    assert result == 25.0 + 273.15  # 298.15 K, the in-window degC peak converted to K
    assert 250.0 <= result <= 330.0  # Kelvin band, never ~50-110 °F


async def test_fetch_returns_kelvin_high_in_window():
    captured: dict = {}
    client = _build_mock_client(captured)
    try:
        result = await fetch_nws_forecast(CITY, TARGET, client=client)
    finally:
        await client.aclose()
    assert result is not None
    # 30 C in-window peak -> 303.15 K; the 45 C value on the day before must NOT win.
    assert result == 30.0 + 273.15
    assert 250.0 <= result <= 330.0


async def test_every_request_carries_user_agent():
    captured: dict = {}
    client = _build_mock_client(captured)
    try:
        await fetch_nws_forecast(CITY, TARGET, client=client)
    finally:
        await client.aclose()
    assert captured["user_agents"], "expected at least one request"
    assert all(ua for ua in captured["user_agents"]), (
        "every api.weather.gov request must carry a User-Agent (Pitfall 7)"
    )


def test_store_uses_nws_label_member_zero(monkeypatch):
    win = settlement_window(get_city(CITY), TARGET)
    captured: dict = {}

    def fake_insert(bind, **kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr(nws_mod, "insert_forecast", fake_insert)
    rc = nws_mod.store_nws_forecast(object(), CITY, TARGET, 303.15)
    assert rc == 1
    assert captured["model"] == MODEL == "nws"
    assert captured["member"] == 0
    assert 250.0 <= captured["temp_kelvin"] <= 330.0
    assert captured["station_lat"] == get_city(CITY).lat
    assert captured["available_at"].tzinfo is not None  # tz-aware (live now())
    _ = win
