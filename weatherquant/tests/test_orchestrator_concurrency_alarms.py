"""WR-03 / WR-05: off-loop AFD classify + correctness alarms are not swallowed (D-14/D-11).

* WR-03 (D-14) — ``classify_afd`` may issue a BLOCKING OpenAI SDK call; ``ingest_afd`` must
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
from weatherquant.ingest.errors import (
    CorrectnessError,
    SanityError,
    TargetDateError,
    UnitError,
)
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

    # These tests exercise the WRITER/snap alarm-propagation contract at lead=0, not the lead-0
    # ASOS probe — stub it to None (skip) so the probe never makes a real network call.
    async def _no_asos(*_a, **_kw):  # noqa: ANN001
        return None

    monkeypatch.setattr(orchestrator.obs, "asos_lead0_kelvin", _no_asos)


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


# --- WR-05 FULL: the bare-ValueError correctness alarms now propagate (the partial gap) ------


async def test_unit_error_propagates_not_swallowed(
    grib_alarm_env, monkeypatch: pytest.MonkeyPatch
):
    """WR-05 (full): a UnitError (unit mismatch) PROPAGATES — previously a bare ValueError swallowed.

    A UnitError IS-A ValueError, so the old ``except Exception`` would have downgraded it to a
    silent "missing cycle" skip. It must now fail loud (it is a CorrectnessError).
    """

    def _raise_unit(_bind, **_kw):  # noqa: ANN001
        raise UnitError("decoded TMP:2 m units must be 'K'; got units='F'")

    monkeypatch.setattr(orchestrator, "insert_forecast", _raise_unit)

    with pytest.raises(UnitError):
        await orchestrator.ingest_cycle(object(), "nbm", "NYC", _CYCLE, mode="backfill", lead=0)


async def test_sanity_error_propagates_not_swallowed(monkeypatch: pytest.MonkeyPatch):
    """WR-05 (full): a SanityError (lead-0 / snap breach) raised during the GRIB path PROPAGATES."""

    fake_field = object()

    def _fake_fetch_t2m(model, cycle_init, fxx, member="c00"):  # noqa: ANN001
        return fake_field

    def _snap_breach(_field, _city, **_kw):  # noqa: ANN001
        # The snap raises a SanityError (out-of-domain station / wrong grid) — a correctness
        # alarm, NOT a transient absence. It is a ValueError too, so the old catch swallowed it.
        raise SanityError("nearest grid point 9e9 m from station exceeds bound (Pitfall 2)")

    monkeypatch.setattr(grib, "fetch_t2m", _fake_fetch_t2m)
    monkeypatch.setattr(grib, "snap_city", _snap_breach)

    with pytest.raises(SanityError):
        await orchestrator.ingest_cycle(object(), "nbm", "NYC", _CYCLE, mode="backfill", lead=0)


# --- NEW-1: _target_date_for's fail-loud raise now actually escapes ingest_cycle ------------


async def test_target_date_error_propagates_out_of_ingest_cycle(monkeypatch: pytest.MonkeyPatch):
    """NEW-1: an impossible settlement window raises TargetDateError OUT of ingest_cycle.

    ``_target_date_for`` runs INSIDE ingest_cycle's try; the regression was that its raise was a
    bare ValueError, caught and downgraded to a silent skip — neutralizing WR-04 for every
    ingest path. With TargetDateError (a CorrectnessError) the fail-loud guard actually escapes.
    """

    def _raise_target_date(_city, _cycle, _lead):  # noqa: ANN001
        raise TargetDateError("no settlement window contains valid instant ... (D-16)")

    monkeypatch.setattr(orchestrator, "_target_date_for", _raise_target_date)

    with pytest.raises(TargetDateError):
        await orchestrator.ingest_cycle(object(), "nbm", "NYC", _CYCLE, mode="backfill", lead=0)


def test_all_alarms_are_correctness_errors():
    """Source guard: every alarm type the orchestrator must re-raise IS-A CorrectnessError."""
    for exc in (WriteIntegrityError, UnitError, SanityError, TargetDateError):
        assert issubclass(exc, CorrectnessError), exc.__name__
    # And the ValueError-compatible ones preserve the legacy contract.
    for exc in (UnitError, SanityError, TargetDateError):
        assert issubclass(exc, ValueError), exc.__name__
