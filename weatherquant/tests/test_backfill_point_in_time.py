"""CR-01 / WR-01 / WR-02 / WR-04: point-in-time integrity on the backfill path (D-09/D-11).

The whole ingestion spine exists to keep ``available_at`` honest: a backfilled (historical)
row must NEVER be stamped with the wall clock, or it appears "unavailable until the backfill
ran" and silently corrupts Phase 6's no-look-ahead walk-forward (D-09). These tests pin the
backfill seam:

* CR-01 — AFD never stamps now() in backfill. ``store_afd_signal`` raises rather than
  defaulting to now() when no issuance time is supplied; the orchestrator SKIPS AFD in plain
  backfill (absence = absence) instead of fabricating a wall-clock row.
* WR-01 — the supplementary ``store_*`` functions thread ``mode`` into ``available_at`` (not a
  hardcoded "live"), so backfill stamps cycle+latency / live stamps now() through the ONE seam.
* WR-04 — ``_target_date_for`` RAISES on an impossible window instead of silently returning a
  hand-rolled UTC date (the v3 anti-pattern, D-16).

Offline: no DB, no network. The audited writer is patched to a recorder so the stamped
``available_at`` is inspected directly.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone

import pytest

from weatherquant.ingest import afd as afd_mod
from weatherquant.ingest import orchestrator
from weatherquant.ingest.afd import store_afd_signal
from weatherquant.ingest.sources import nws, openmeteo, wethr

_CYCLE = datetime(2024, 1, 15, 0, tzinfo=timezone.utc)
_TARGET = date(2024, 1, 15)


class _Recorder:
    def __init__(self) -> None:
        self.rows: list[dict] = []


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()

    def _fake_insert_forecast(_bind, **kwargs):  # noqa: ANN001
        rec.rows.append(kwargs)
        return 1

    def _fake_insert_observation(_bind, **kwargs):  # noqa: ANN001
        rec.rows.append(kwargs)
        return 1

    for mod in (nws, openmeteo, wethr):
        monkeypatch.setattr(mod, "insert_forecast", _fake_insert_forecast)
    monkeypatch.setattr(afd_mod, "insert_observation", _fake_insert_observation)
    return rec


# --- CR-01: AFD never stamps now() in backfill ------------------------------------------


def test_store_afd_signal_backfill_without_issuance_raises_not_now(recorder: _Recorder):
    """CR-01: backfill AFD with no issuance time RAISES — never silently stamps now()."""
    with pytest.raises(ValueError, match="refusing to stamp now"):
        store_afd_signal(
            object(), "NYC", _TARGET, {"disagreement": False}, available_at=None, mode="backfill"
        )
    assert recorder.rows == []  # nothing was written


def test_store_afd_signal_backfill_with_issuance_stamps_that_instant(recorder: _Recorder):
    """CR-01: an explicit issuance time IS used verbatim in backfill (the report time, D-09)."""
    issuance = datetime(2024, 1, 15, 3, 47, tzinfo=timezone.utc)
    store_afd_signal(
        object(), "NYC", _TARGET, {"disagreement": True, "direction": "warmer"},
        available_at=issuance, mode="backfill",
    )
    assert len(recorder.rows) == 1
    assert recorder.rows[0]["available_at"] == issuance


def test_store_afd_signal_live_defaults_to_now(recorder: _Recorder):
    """Live mode keeps the now() default (the instant the running system held the product)."""
    before = datetime.now(timezone.utc)
    store_afd_signal(object(), "NYC", _TARGET, {"disagreement": False}, mode="live")
    after = datetime.now(timezone.utc)
    assert len(recorder.rows) == 1
    assert before <= recorder.rows[0]["available_at"] <= after


async def test_ingest_afd_backfill_skips_with_structured_log(
    recorder: _Recorder, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """CR-01: ingest_afd in backfill (no issuance) SKIPS + logs — no fetch, no now() row."""
    fetched = {"called": False}

    async def _fail_fetch(_wfo, **_kw):  # noqa: ANN001
        fetched["called"] = True
        return "some afd text with model disagreement"

    monkeypatch.setattr(afd_mod, "fetch_afd_text", _fail_fetch)

    with caplog.at_level(logging.INFO):
        n = await orchestrator.ingest_afd(object(), "NYC", _TARGET, mode="backfill")

    assert n == 0
    assert recorder.rows == []  # absence = absence — no fabricated now() row
    assert not fetched["called"]  # skipped BEFORE any (paid) fetch/classify
    assert any("live-only" in r.message and "source=afd" in r.message for r in caplog.records)


# --- WR-01: supplementary store_* thread mode into available_at -------------------------


def test_store_nws_forecast_backfill_stamps_now_via_live_branch(recorder: _Recorder):
    # The nws label has no PUBLISH_LATENCY entry; backfill is not run for it (WR-02), but the
    # seam must still thread mode. Live stamps now(); we assert the seam accepts mode at all.
    before = datetime.now(timezone.utc)
    nws.store_nws_forecast(object(), "NYC", _TARGET, 290.0, cycle=_CYCLE, mode="live")
    after = datetime.now(timezone.utc)
    assert before <= recorder.rows[0]["available_at"] <= after


def test_store_openmeteo_threads_mode_into_available_at(recorder: _Recorder):
    # openmeteo control label = "openmeteo" (no latency entry) — assert mode is threaded, i.e.
    # live still stamps now() (and the call accepts the mode kwarg, proving the seam, WR-01).
    before = datetime.now(timezone.utc)
    openmeteo.store_members(object(), "NYC", _TARGET, {0: 291.0}, cycle=_CYCLE, mode="live")
    after = datetime.now(timezone.utc)
    assert before <= recorder.rows[0]["available_at"] <= after


def test_store_wethr_threads_mode_into_available_at(recorder: _Recorder):
    before = datetime.now(timezone.utc)
    wethr.store_wethr_forecast(object(), "NYC", "hrrr", _TARGET, 289.0, cycle=_CYCLE, mode="live")
    after = datetime.now(timezone.utc)
    assert before <= recorder.rows[0]["available_at"] <= after


def test_store_signatures_accept_mode_kwarg():
    """WR-01 source guard: every supplementary store_* exposes a ``mode`` parameter."""
    import inspect

    for fn in (nws.store_nws_forecast, openmeteo.store_members, wethr.store_wethr_forecast):
        assert "mode" in inspect.signature(fn).parameters, fn.__name__


# --- WR-04: _target_date_for raises instead of a silent wrong-date fallback -------------


def test_target_date_for_resolves_normal_offset():
    # For a normal US offset the settlement-window loop always matches (sanity baseline).
    td = orchestrator._target_date_for("NYC", _CYCLE, 12)
    assert isinstance(td, date)


def test_target_date_for_raises_on_impossible_window(monkeypatch: pytest.MonkeyPatch):
    """WR-04: when no candidate window contains the valid instant, RAISE (no UTC fallback)."""
    from weatherquant.ingest import orchestrator as orch
    from weatherquant.time import SettlementWindow

    # Force every candidate window to NOT contain the valid instant (a broken-offset stand-in).
    def _empty_window(_city, candidate):  # noqa: ANN001
        far = datetime(1900, 1, 1, tzinfo=timezone.utc)
        return SettlementWindow(
            local_date=candidate,
            start_utc=far,
            end_utc=far + timedelta(seconds=1),
            std_offset_hours=-5,
            station="KNYC",
        )

    monkeypatch.setattr(orch, "settlement_window", _empty_window)
    # NEW-1: the real raise is a TargetDateError (a CorrectnessError) so it escapes the
    # ingest_cycle catch — and still a ValueError so this legacy contract holds.
    from weatherquant.ingest.errors import CorrectnessError, TargetDateError

    with pytest.raises(TargetDateError, match="no settlement window contains valid instant"):
        orch._target_date_for("NYC", _CYCLE, 0)
    assert issubclass(TargetDateError, CorrectnessError)
    assert issubclass(TargetDateError, ValueError)


# --- asos-rate-limit-empty-obs: a rate-limited obs fetch must NOT fabricate an empty label ----


async def test_ingest_obs_rate_limited_fetch_writes_no_empty_label(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """A persistent 429 (ObsFetchError) must SKIP the day — never persist daily_high_f=NULL,obs_count=0."""
    from weatherquant.ingest import obs as obs_mod
    from weatherquant.ingest.errors import ObsFetchError

    written: list[dict] = []

    def _record_insert(_bind, **kwargs):  # noqa: ANN001
        written.append(kwargs)
        return 1

    async def _rate_limited(_city, _win, *_a, **_kw):  # noqa: ANN001
        raise ObsFetchError("IEM ASOS rate-limited (429) after retries — fallback unavailable")

    # Patch the writer the obs path routes through and force the fetch to fail-loud.
    monkeypatch.setattr(obs_mod, "insert_observation", _record_insert)
    monkeypatch.setattr(orchestrator.obs, "fetch_asos_obs", _rate_limited)

    with caplog.at_level(logging.WARNING):
        n = await orchestrator.ingest_obs(object(), "CHI", date(2025, 3, 1))

    assert n == 0  # day skipped
    assert written == []  # NO fabricated empty ground-truth row persisted
    # ObsFetchError is a RuntimeError (not a CorrectnessError) → graceful per-day degrade, not abort.
    assert any("source=asos" in r.message for r in caplog.records)


def test_obs_fetch_error_is_not_a_correctness_error():
    """ObsFetchError must be a plain RuntimeError so the orchestrator degrades (skips) rather than aborts."""
    from weatherquant.ingest.errors import CorrectnessError, ObsFetchError

    assert issubclass(ObsFetchError, RuntimeError)
    assert not issubclass(ObsFetchError, CorrectnessError)
