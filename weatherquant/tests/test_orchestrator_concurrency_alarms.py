"""WR-03 / WR-05: off-loop AFD classify + correctness alarms are not swallowed (D-14/D-11).

* WR-03 (D-14) — ``classify_afd`` may issue a BLOCKING Anthropic SDK call; ``ingest_afd`` must
  run it in a thread executor (like the GRIB decode) so it never blocks the async loop. We
  assert classify runs on a DIFFERENT thread than the event loop.
* WR-05 (D-11) — the per-source ``except`` is graceful degradation for EXPECTED transient
  failures only. Correctness ALARMS (the writer's ``WriteIntegrityError``, an ``AssertionError``)
  must PROPAGATE, not be downgraded to a silent "missing cycle" skip. A transient error
  (e.g. an HTTP/decode failure) still degrades gracefully.

Offline: no DB, no network, the SDK/writer are stubbed.
"""

from __future__ import annotations

import threading
from datetime import date, datetime, timezone

import pytest

from weatherquant.ingest import afd as afd_mod
from weatherquant.ingest import grib, orchestrator
from weatherquant.ingest.writer import WriteIntegrityError

_CYCLE = datetime(2026, 6, 12, 0, tzinfo=timezone.utc)
_TARGET = date(2026, 6, 12)


# --- WR-03: classify_afd runs OFF the event loop ----------------------------------------


async def test_classify_afd_runs_off_event_loop(monkeypatch: pytest.MonkeyPatch):
    """WR-03/D-14: the (blocking) classify call runs in a thread executor, not on the loop."""
    main_thread = threading.current_thread().ident
    seen: dict = {}

    async def _fake_fetch(_wfo, **_kw):  # noqa: ANN001
        return "afd text with model disagreement"

    def _fake_classify(_text, _wfo, *a, **k):  # noqa: ANN001
        # Record the thread classify ran on — it must NOT be the event-loop thread (WR-03).
        seen["thread"] = threading.current_thread().ident
        return {"disagreement": True, "direction": "warmer", "summary": "x"}

    def _fake_store(_bind, _city, _td, _sig, **_kw):  # noqa: ANN001
        return 1

    monkeypatch.setattr(afd_mod, "fetch_afd_text", _fake_fetch)
    monkeypatch.setattr(afd_mod, "classify_afd", _fake_classify)
    monkeypatch.setattr(afd_mod, "store_afd_signal", _fake_store)

    # Live mode (so AFD actually runs end-to-end; backfill would skip it — CR-01).
    n = await orchestrator.ingest_afd(object(), "NYC", _TARGET, mode="live")
    assert n == 1
    assert "thread" in seen
    assert seen["thread"] != main_thread, "classify_afd must run off the event-loop thread"


# --- WR-05: correctness alarms propagate; transient errors degrade ----------------------


@pytest.fixture
def grib_alarm_env(monkeypatch: pytest.MonkeyPatch):
    """Make the GRIB path fetch+snap OK but the audited writer raise a WriteIntegrityError."""

    fake_field = object()

    def _fake_fetch_t2m(model, cycle_init, fxx, member="c00"):  # noqa: ANN001
        return fake_field

    def _fake_snap_city(_field, _city, **_kw):  # noqa: ANN001
        return 295.0, 40.779, -73.969, 1234.0

    monkeypatch.setattr(grib, "fetch_t2m", _fake_fetch_t2m)
    monkeypatch.setattr(grib, "snap_city", _fake_snap_city)


async def test_write_integrity_error_propagates_not_swallowed(
    grib_alarm_env, monkeypatch: pytest.MonkeyPatch
):
    """WR-05: a WriteIntegrityError (a row that should have landed didn't) PROPAGATES."""

    def _raise_integrity(_bind, **_kw):  # noqa: ANN001
        raise WriteIntegrityError("expected rowcount==1 inserting into forecasts, got 0")

    monkeypatch.setattr(orchestrator, "insert_forecast", _raise_integrity)

    with pytest.raises(WriteIntegrityError):
        await orchestrator.ingest_cycle(object(), "nbm", "NYC", _CYCLE, mode="backfill", lead=0)


async def test_assertion_error_propagates_not_swallowed(
    grib_alarm_env, monkeypatch: pytest.MonkeyPatch
):
    """WR-05: an AssertionError (a sanity contract) PROPAGATES rather than degrading."""

    def _raise_assert(_bind, **_kw):  # noqa: ANN001
        raise AssertionError("lead-0 sanity contract")

    monkeypatch.setattr(orchestrator, "insert_forecast", _raise_assert)

    with pytest.raises(AssertionError):
        await orchestrator.ingest_cycle(object(), "nbm", "NYC", _CYCLE, mode="backfill", lead=0)


async def test_transient_runtimeerror_still_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """WR-05: a transient fetch RuntimeError (late/missing cycle) STILL degrades (returns 0)."""
    import logging

    def _raise_missing(model, cycle_init, fxx, member="c00"):  # noqa: ANN001
        raise RuntimeError("HRRR cycle not yet published (late/missing)")

    monkeypatch.setattr(grib, "fetch_t2m", _raise_missing)

    with caplog.at_level(logging.WARNING):
        n = await orchestrator.ingest_cycle(object(), "hrrr", "NYC", _CYCLE, mode="backfill", lead=0)
    assert n == 0  # graceful skip
    assert any("ingest fallback" in r.message and "source=hrrr" in r.message for r in caplog.records)
