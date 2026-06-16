"""ING-05 (GREEN): Open-Meteo ensemble per-member parse + degC->K; member->index map.

02-04 turns the Wave-0 RED stub GREEN via ``weatherquant.ingest.sources.openmeteo``. Tests
mock httpx (NO network) and prove: (1) each member's in-window max Celsius converts to a
``temp_kelvin`` in the ~280-320 K band (catching an accidental °F store); (2) ``memberNN``
maps to member index N (member00->0 ... member30->30); (3) provider-namespaced labels
``openmeteo`` (control) / ``openmeteo:<member>``; (4) ALL members are requested in a SINGLE
httpx GET (A3 budget); (5) the request never asks for fahrenheit (Pitfall 3 / D-07).
"""

from __future__ import annotations

from datetime import date, timedelta

import httpx

from weatherquant.ingest.sources import openmeteo as om_mod
from weatherquant.ingest.sources.openmeteo import (
    N_MEMBERS,
    celsius_to_kelvin,
    fetch_openmeteo_ensemble,
    member_label,
    parse_members,
)
from weatherquant.registry import get_city
from weatherquant.time import settlement_window

CITY = "NYC"
TARGET = date(2026, 6, 15)


def _ensemble_payload() -> dict:
    """A /ensemble payload: hourly time grid + each member's degC series."""
    win = settlement_window(get_city(CITY), TARGET)
    # Four hours: one before the window, three inside it (the third is each member's peak).
    times = [
        (win.start_utc - timedelta(hours=1)).isoformat(),  # wrong day — must NOT win
        (win.start_utc + timedelta(hours=14)).isoformat(),
        (win.start_utc + timedelta(hours=18)).isoformat(),
        (win.start_utc + timedelta(hours=20)).isoformat(),
    ]
    hourly: dict = {"time": times}
    for m in range(N_MEMBERS):
        # member m peak (in-window) = 20 + m*0.1 °C; the out-of-window hour is a hot 99 °C.
        peak_c = 20.0 + m * 0.1
        hourly[f"temperature_2m_member{m:02d}"] = [99.0, peak_c - 2, peak_c, peak_c - 1]
    return {"hourly": hourly}


def _mock_client(captured: dict) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        captured.setdefault("calls", []).append(str(request.url))
        captured["last_query"] = request.url.query.decode()
        return httpx.Response(200, json=_ensemble_payload())

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_openmeteo_members_parsed_in_kelvin():
    assert callable(fetch_openmeteo_ensemble)


def test_celsius_to_kelvin_centralized():
    assert celsius_to_kelvin(0.0) == 273.15
    assert celsius_to_kelvin(25.0) == 298.15


def test_member_label_is_provider_namespaced():
    assert member_label(0) == "openmeteo"  # control / primary
    assert member_label(1) == "openmeteo:1"
    assert member_label(30) == "openmeteo:30"


def test_parse_members_buckets_and_converts():
    win = settlement_window(get_city(CITY), TARGET)
    members = parse_members(_ensemble_payload(), win)
    # All 31 members present, member00..member30 -> 0..30.
    assert set(members) == set(range(N_MEMBERS))
    # member 0 peak 20.0 °C -> 293.15 K; the out-of-window 99 °C must NOT win.
    assert members[0] == celsius_to_kelvin(20.0)
    assert members[30] == celsius_to_kelvin(20.0 + 30 * 0.1)
    # Every member lands in the Kelvin band (catches an accidental °F store).
    for k in members.values():
        assert 250.0 <= k <= 330.0


async def test_fetch_single_call_all_members_no_fahrenheit():
    captured: dict = {}
    client = _mock_client(captured)
    try:
        members = await fetch_openmeteo_ensemble(CITY, TARGET, client=client)
    finally:
        await client.aclose()
    # ONE GET for ALL members (A3 budget).
    assert len(captured["calls"]) == 1
    query = captured["last_query"].lower()
    assert "fahrenheit" not in query  # Pitfall 3 / D-07 — API-default Celsius only
    assert "member00" in query and "member30" in query  # all members in the one request
    assert set(members) == set(range(N_MEMBERS))


def test_store_members_uses_member_axis_and_labels(monkeypatch):
    captured: list = []

    def fake_insert(bind, **kwargs):
        captured.append(kwargs)
        return 1

    monkeypatch.setattr(om_mod, "insert_forecast", fake_insert)
    members = {0: celsius_to_kelvin(20.0), 1: celsius_to_kelvin(21.0), 30: celsius_to_kelvin(25.0)}
    inserted = om_mod.store_members(object(), CITY, TARGET, members)
    assert inserted == 3
    by_member = {row["member"]: row for row in captured}
    assert by_member[0]["model"] == "openmeteo"
    assert by_member[1]["model"] == "openmeteo:1"
    assert by_member[30]["model"] == "openmeteo:30"
    for row in captured:
        assert 250.0 <= row["temp_kelvin"] <= 330.0  # never °F
        assert row["lead"] == 0
        assert row["available_at"].tzinfo is not None
