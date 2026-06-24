"""ING-03: daily-high = max(tmpf) over the half-open LST ``settlement_window``.

GREEN (02-03). The daily high MUST be bucketed through
``weatherquant.time.settlement_window`` ([start, end), half-open — D-16), never a hand-rolled
UTC day. These tests prove: (1) the result exposes the stub-contract attributes; (2) the
boundary is exclusive (a reading at ``end_utc`` does NOT count); (3) a hotter reading just
OUTSIDE the window does NOT raise the daily high (correct bucketing, not a flat window);
(4) ``obs_count`` equals the in-window count; (5) the °F conversion is centralized; (6) a
CLI disagreement is flagged but the ASOS label is still produced (never overwritten — D-16).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import httpx
import pytest

from weatherquant.ingest import obs as obs_mod
from weatherquant.ingest.errors import ObsFetchError
from weatherquant.ingest.obs import (
    DailyHigh,
    celsius_to_fahrenheit,
    daily_high,
    daily_high_from_obs,
    fetch_asos_obs,
)
from weatherquant.registry import get_city
from weatherquant.time import settlement_window


def test_stub_contract_result_shape():
    # RED-stub contract: daily_high_from_obs(city, target_date, readings) -> has the attrs.
    result = daily_high_from_obs(city="NYC", target_date=date(2025, 1, 15), readings=[])
    assert hasattr(result, "daily_high_f")
    assert hasattr(result, "obs_count")
    assert isinstance(result, DailyHigh)
    assert result.daily_high_f is None  # no readings → no label
    assert result.obs_count == 0


def test_daily_high_is_max_over_settlement_window():
    target = date(2025, 1, 15)
    win = settlement_window(get_city("NYC"), target)
    # Three in-window readings; the max is the label.
    rows = [
        (win.start_utc + timedelta(hours=2), 41.0),
        (win.start_utc + timedelta(hours=9), 55.0),  # the peak
        (win.start_utc + timedelta(hours=14), 48.0),
    ]
    result = daily_high(rows, "NYC", target)
    assert result.daily_high_f == 55.0
    assert result.obs_count == 3
    assert result.window_start == win.start_utc
    assert result.window_end == win.end_utc
    # available-at provenance is the report time of the peak reading (D-09), not now().
    assert result.report_time == win.start_utc + timedelta(hours=9)


def test_boundary_end_utc_is_excluded_half_open():
    target = date(2025, 1, 15)
    win = settlement_window(get_city("NYC"), target)
    rows = [
        (win.start_utc, 30.0),  # start is INCLUSIVE
        (win.end_utc - timedelta(minutes=1), 50.0),  # just before end → included
        (win.end_utc, 99.0),  # exactly end_utc → EXCLUDED (half-open)
    ]
    result = daily_high(rows, "NYC", target)
    assert result.obs_count == 2  # start + just-before-end, NOT the end_utc reading
    assert result.daily_high_f == 50.0  # the 99 at end_utc must not win


def test_hotter_reading_just_outside_window_does_not_win():
    # The v3 flat-window bug: a hotter reading from the wrong LST day raises the high.
    target = date(2025, 1, 15)
    win = settlement_window(get_city("NYC"), target)
    rows = [
        (win.start_utc + timedelta(hours=10), 52.0),  # in-window true high
        (win.start_utc - timedelta(hours=1), 80.0),  # hotter, BEFORE the window
        (win.end_utc + timedelta(hours=1), 90.0),  # hotter, AFTER the window
    ]
    result = daily_high(rows, "NYC", target)
    assert result.daily_high_f == 52.0  # the outside-window 80/90 do NOT win
    assert result.obs_count == 1


def test_obs_count_equals_in_window_rows():
    target = date(2025, 1, 15)
    win = settlement_window(get_city("CHI"), target)
    in_window = [
        (win.start_utc + timedelta(hours=h), 30.0 + h) for h in range(6)
    ]
    out_window = [(win.end_utc + timedelta(hours=2), 99.0)]
    result = daily_high(in_window + out_window, "CHI", target)
    assert result.obs_count == 6
    assert result.daily_high_f == 35.0  # 30 + 5


def test_celsius_to_fahrenheit_centralized():
    assert celsius_to_fahrenheit(0.0) == 32.0
    assert celsius_to_fahrenheit(100.0) == 212.0
    assert celsius_to_fahrenheit(37.0) == pytest.approx(98.6)


def test_cli_disagreement_flags_but_does_not_overwrite_label():
    # D-16: ASOS-max vs CLI disagreement is a flagged event; the ASOS label still stands.
    target = date(2025, 1, 15)
    win = settlement_window(get_city("NYC"), target)
    rows = [(win.start_utc + timedelta(hours=8), 60.0)]
    # CLI oracle says 41 — a >1.5°F disagreement.
    result = daily_high(rows, "NYC", target, cli_max_f=41.0)
    assert result.cli_disagreement is True
    assert result.daily_high_f == 60.0  # ASOS label is NOT overwritten by the CLI max
    assert result.cli_max_f == 41.0


def test_cli_agreement_within_tolerance_not_flagged():
    target = date(2025, 1, 15)
    win = settlement_window(get_city("NYC"), target)
    rows = [(win.start_utc + timedelta(hours=8), 41.4)]
    result = daily_high(rows, "NYC", target, cli_max_f=41.0)  # within 1.5°F
    assert result.cli_disagreement is False


def test_malformed_rows_are_skipped_not_stored():
    target = date(2025, 1, 15)
    win = settlement_window(get_city("NYC"), target)
    rows = [
        (win.start_utc + timedelta(hours=3), 45.0),  # good
        ("not-a-timestamp", 99.0),  # bad ts
        (win.start_utc + timedelta(hours=4), None),  # bad temp
        (win.start_utc + timedelta(hours=5),),  # too short
        {"ts_utc": (win.start_utc + timedelta(hours=6)).isoformat(), "temp_f": 50.0},  # mapping
    ]
    result = daily_high(rows, "NYC", target)
    assert result.obs_count == 2  # only the two well-formed readings
    assert result.daily_high_f == 50.0


def test_naive_timestamps_assumed_utc():
    target = date(2025, 1, 15)
    win = settlement_window(get_city("NYC"), target)
    naive = (win.start_utc + timedelta(hours=7)).replace(tzinfo=None)
    result = daily_high([(naive, 47.0)], "NYC", target)
    assert result.obs_count == 1
    assert result.daily_high_f == 47.0


# --- fetch_asos_obs IEM -> AWC fallback gating (Phase-4 fix #3) ----------------------------


def _routing_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_empty_iem_success_does_not_trigger_awc_fallback():
    """A successful-but-EMPTY IEM response is a real 'no obs', not a failure — no AWC fallback."""
    win = settlement_window(get_city("NYC"), datetime.now(UTC).date())
    calls = {"iem": 0, "awc": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "mesonet" in request.url.host:
            calls["iem"] += 1
            return httpx.Response(200, text="station,valid,tmpf\n")  # header only, no data rows
        calls["awc"] += 1
        return httpx.Response(200, json=[])

    client = _routing_client(handler)
    try:
        rows = await fetch_asos_obs("NYC", win, client=client)
    finally:
        await client.aclose()
    assert rows == []
    assert calls["iem"] == 1
    assert calls["awc"] == 0  # empty-but-successful must NOT trigger the live fallback


async def test_iem_failure_falls_back_to_awc_for_live_window():
    """On a real IEM failure with a live/today window, the AWC fallback IS invoked."""
    win = settlement_window(get_city("NYC"), datetime.now(UTC).date())
    calls = {"awc": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "mesonet" in request.url.host:
            return httpx.Response(500, text="boom")
        calls["awc"] += 1
        return httpx.Response(200, json=[])

    client = _routing_client(handler)
    try:
        rows = await fetch_asos_obs("NYC", win, client=client)
    finally:
        await client.aclose()
    assert calls["awc"] == 1
    assert isinstance(rows, list)


async def test_awc_fallback_skipped_for_historical_backfill_window():
    """A historical (backfill) window predates the recent ~14h AWC feed — skip it (no leak)."""
    win = settlement_window(get_city("NYC"), date(2025, 1, 15))
    calls = {"awc": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "mesonet" in request.url.host:
            return httpx.Response(500, text="boom")  # IEM fails → would otherwise trigger fallback
        calls["awc"] += 1
        return httpx.Response(200, json=[{"temp": 5.0, "reportTime": "2026-06-21T12:00:00Z"}])

    client = _routing_client(handler)
    try:
        rows = await fetch_asos_obs("NYC", win, client=client)
    finally:
        await client.aclose()
    assert rows == []
    assert calls["awc"] == 0  # gated out before any HTTP call — no wrong-date obs fetched


# --- IEM 429 retry/backoff + no-fabricated-empty-label (asos-rate-limit-empty-obs fix) -------


@pytest.fixture
def no_sleep(monkeypatch):
    """Patch the obs module's asyncio.sleep so backoff retries don't actually wait."""

    async def _instant(_seconds):
        return None

    monkeypatch.setattr(obs_mod.asyncio, "sleep", _instant)


async def test_iem_429_then_success_retries_and_returns_rows(no_sleep):
    """A 429 on the first attempt is retried with backoff; a subsequent 200 yields the rows."""
    win = settlement_window(get_city("NYC"), datetime.now(UTC).date())
    valid = (win.start_utc + timedelta(hours=6)).strftime("%Y-%m-%d %H:%M")
    state = {"iem": 0, "awc": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "mesonet" in request.url.host:
            state["iem"] += 1
            if state["iem"] == 1:
                return httpx.Response(429, headers={"Retry-After": "1"}, text="rate limited")
            return httpx.Response(200, text=f"station,valid,tmpf\nNYC,{valid},55.0\n")
        state["awc"] += 1
        return httpx.Response(200, json=[])

    client = _routing_client(handler)
    try:
        rows = await fetch_asos_obs("NYC", win, client=client)
    finally:
        await client.aclose()
    assert state["iem"] == 2  # one 429 + one success
    assert state["awc"] == 0  # success on retry → fallback never touched
    assert rows == [(datetime.fromisoformat(valid).replace(tzinfo=UTC), 55.0)]


async def test_persistent_429_backfill_raises_not_empty_label(no_sleep):
    """Retries exhausted on a 429 over a backfill window must RAISE, never return [] (no empty row)."""
    win = settlement_window(get_city("NYC"), date(2025, 1, 15))  # historical → AWC gated out
    state = {"iem": 0, "awc": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "mesonet" in request.url.host:
            state["iem"] += 1
            return httpx.Response(429, text="rate limited")
        state["awc"] += 1
        return httpx.Response(200, json=[{"temp": 5.0, "reportTime": "2026-06-21T12:00:00Z"}])

    client = _routing_client(handler)
    try:
        with pytest.raises(ObsFetchError):
            await fetch_asos_obs("NYC", win, client=client)
    finally:
        await client.aclose()
    # Every attempt 429'd: initial + _IEM_429_MAX_RETRIES retries.
    assert state["iem"] == obs_mod._IEM_429_MAX_RETRIES + 1
    assert state["awc"] == 0  # backfill window gates the live fallback out before any HTTP call


async def test_persistent_429_live_window_recovers_via_awc(no_sleep):
    """Retries exhausted on a 429 over a LIVE window recover through the AWC fallback (no raise)."""
    win = settlement_window(get_city("NYC"), datetime.now(UTC).date())
    in_window_ts = (win.start_utc + timedelta(hours=2)).isoformat()
    state = {"awc": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "mesonet" in request.url.host:
            return httpx.Response(429, text="rate limited")
        state["awc"] += 1
        return httpx.Response(200, json=[{"temp": 10.0, "reportTime": in_window_ts}])

    client = _routing_client(handler)
    try:
        rows = await fetch_asos_obs("NYC", win, client=client)
    finally:
        await client.aclose()
    assert state["awc"] == 1
    assert rows == [(datetime.fromisoformat(in_window_ts), celsius_to_fahrenheit(10.0))]


async def test_retry_after_header_honored_for_backoff(no_sleep, monkeypatch):
    """The backoff delay honors a numeric Retry-After header (capped)."""
    delays: list[float] = []

    async def _record(seconds):
        delays.append(seconds)

    monkeypatch.setattr(obs_mod.asyncio, "sleep", _record)
    win = settlement_window(get_city("NYC"), datetime.now(UTC).date())
    state = {"iem": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["iem"] += 1
        if state["iem"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"}, text="rate limited")
        return httpx.Response(200, text="station,valid,tmpf\n")

    client = _routing_client(handler)
    try:
        await fetch_asos_obs("NYC", win, client=client)
    finally:
        await client.aclose()
    assert delays == [7.0]  # the Retry-After value, not the exponential default


def test_cli_fixture_parity_window_max(cli_fixture):
    # Each fixture day's in-window max must equal its CLI max, and the just-out-of-window
    # hotter reading must NOT win (the conftest guarantees both per day).
    for code, payload in cli_fixture.items():
        for _season, day in payload["days"].items():
            target = date.fromisoformat(day["date"])
            rows = [
                (datetime.fromisoformat(o["ts_utc"].replace("Z", "+00:00")), o["temp_f"])
                for o in day["obs"]
            ]
            result = daily_high(rows, code, target)
            assert result.daily_high_f == day["cli_max"], (
                f"{code}/{day['date']}: window-max {result.daily_high_f} != "
                f"CLI {day['cli_max']}"
            )
