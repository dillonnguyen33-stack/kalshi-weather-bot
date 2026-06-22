"""ING-08 / D-11: a missing/raising model cycle is logged + skipped, others still ingest.

Turned GREEN by 02-05's orchestrator (the one ingestion code path). The behavioral
contract under test:

* When ONE model's fetch raises (a late/missing cycle), a STRUCTURED fallback is logged and
  the OTHER models still ingest their rows — the city/cycle is NOT dropped (D-11).
* NO row is inserted for the failed model: absence is represented by absence — the
  orchestrator never interpolates or carries forward a fake value (D-11). We assert this by
  counting the rows that reached the (mocked) single audited writer path.

The per-source degradation contract lives in ``ingest_cycle``'s try/except; ``_ingest_all``
below drives that ONE code path model-by-model (the trivial fan-out the deleted
``ingest_all_models`` used to wrap). The GRIB fetch + decode and the supplementary HTTP sources
are MOCKED here (no network, no cfgrib, no DB) — this pins the degradation/branching contract,
not the already-unit-tested source parsers (02-02/03/04 cover those).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from weatherquant.ingest import grib, obs, orchestrator
from weatherquant.ingest.sources import nws, openmeteo, wethr


async def _ingest_all(bind, city, cycle, *, mode, lead=0):
    """Drive ``ingest_cycle`` (the ONE code path) over every GRIB model + supplementary source.

    Replaces the deleted ``ingest_all_models``; the degradation/skip logic under test lives in
    ``ingest_cycle`` itself, so fanning out here exercises the same contract.
    """
    targets = [*orchestrator.GRIB_MODELS, *orchestrator.SUPPLEMENTARY_SOURCES]
    return {
        model: await orchestrator.ingest_cycle(
            bind, model, city, cycle, mode=mode, lead=lead
        )
        for model in targets
    }


class _RecordingBind:
    """A fake write target standing in for the SQLAlchemy Engine.

    The orchestrator routes every forecast through ``insert_forecast(bind, ...)``; we patch
    that single audited path to append the row to this recorder instead of touching Postgres.
    Counting recorded rows lets the test assert exactly which models landed a row and which
    (the failed one) landed NONE — proving absence = absence (D-11).
    """

    def __init__(self) -> None:
        self.rows: list[dict] = []


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _RecordingBind:
    """Patch the audited writer + every source fetch so ingestion runs offline.

    * ``insert_forecast`` (the single write path) appends to the recorder and returns 1.
    * GRIB ``fetch_t2m``/``snap_city`` return a fake field/snap so nbm/hrrr/gfs land a row,
      EXCEPT ``hrrr`` whose ``fetch_t2m`` RAISES (the missing/late cycle under test).
    * The supplementary HTTP sources are stubbed to a single deterministic value (nws,
      openmeteo) or skipped (wethr — no key, returns None).
    """
    bind = _RecordingBind()

    def _fake_insert_forecast(_bind, **kwargs):  # noqa: ANN001
        bind.rows.append(kwargs)
        return 1

    # Patch the ONE audited write path everywhere the orchestrator's sources import it.
    monkeypatch.setattr(orchestrator, "insert_forecast", _fake_insert_forecast)
    monkeypatch.setattr(nws, "insert_forecast", _fake_insert_forecast)
    monkeypatch.setattr(openmeteo, "insert_forecast", _fake_insert_forecast)
    monkeypatch.setattr(wethr, "insert_forecast", _fake_insert_forecast)

    # --- GRIB models: nbm/gfs/gefs succeed; hrrr RAISES (the missing-cycle case). ---
    fake_field = object()

    def _fake_fetch_t2m(model, cycle_init, fxx, member="c00"):  # noqa: ANN001
        if model == "hrrr":
            raise RuntimeError("HRRR cycle not yet published (late/missing)")
        return fake_field

    def _fake_snap_city(_field, _city, **_kw):  # noqa: ANN001
        # (temp_kelvin, station_lat, station_lon, grid_distance_m)
        return 295.0, 40.779, -73.969, 1234.0

    monkeypatch.setattr(grib, "fetch_t2m", _fake_fetch_t2m)
    monkeypatch.setattr(grib, "snap_city", _fake_snap_city)

    # --- Supplementary async sources: nws + openmeteo land rows; wethr skips (no key). ---
    async def _fake_nws(_city, _date, **_kw):  # noqa: ANN001
        return 290.0

    async def _fake_openmeteo(_city, _date, **_kw):  # noqa: ANN001
        return {0: 291.0, 1: 292.0}

    async def _fake_wethr(_city, _model, _date, **_kw):  # noqa: ANN001
        return None  # graceful skip — no key (absence = absence)

    monkeypatch.setattr(nws, "fetch_nws_forecast", _fake_nws)
    monkeypatch.setattr(openmeteo, "fetch_openmeteo_ensemble", _fake_openmeteo)
    monkeypatch.setattr(wethr, "fetch_wethr_forecast", _fake_wethr)

    # The lead-0 sanity probe (D-04) would otherwise hit the ASOS feed at lead=0; stub it to
    # None so the probe is skipped and the degradation contract stays the focus here.
    async def _no_asos(*_a, **_kw):  # noqa: ANN001
        return None

    monkeypatch.setattr(obs, "asos_lead0_kelvin", _no_asos)
    return bind


async def test_missing_cycle_does_not_abort_other_models(
    recorder: _RecordingBind, caplog: pytest.LogCaptureFixture
):
    """A raising HRRR cycle degrades gracefully; the other models still ingest (D-11)."""
    cycle = datetime(2026, 6, 12, 0, tzinfo=timezone.utc)
    with caplog.at_level(logging.WARNING):
        summary = await _ingest_all(
            recorder, "NYC", cycle, mode="backfill", lead=0
        )

    # The city/cycle was NOT dropped: the non-failing models all ran and returned a count.
    assert set(summary) == set(orchestrator.GRIB_MODELS) | set(
        orchestrator.SUPPLEMENTARY_SOURCES
    )

    # HRRR raised -> a STRUCTURED fallback was logged and NO hrrr row was inserted.
    assert summary["hrrr"] == 0
    fallback_logs = [r.message for r in caplog.records if "ingest fallback" in r.message]
    assert any("source=hrrr" in m for m in fallback_logs), fallback_logs

    # The OTHER GRIB models still landed exactly one row each (nbm, gfs) + GEFS 31 members.
    assert summary["nbm"] == 1
    assert summary["gfs"] == 1
    assert summary["gefs"] == 31  # c00 + p01..p30

    # WR-02: in BACKFILL the live-only HTTP sources (nws/openmeteo/wethr) are SKIPPED — they
    # return only the current forecast and have no point-in-time historical archive, so they
    # are not run during a historical backfill (absence = absence, D-11). All three report 0.
    assert summary["nws"] == 0
    assert summary["openmeteo"] == 0
    assert summary["wethr"] == 0

    # Absence = absence: NOT A SINGLE recorded row is for the failed hrrr model (D-11).
    assert all(row["model"] != "hrrr" for row in recorder.rows)
    # The recorded row total is the GRIB successes only (nbm + gfs + gefs); the live-only
    # supplementary sources were skipped in backfill, and hrrr failed (no fabricated row).
    assert len(recorder.rows) == 1 + 1 + 31  # nbm + gfs + gefs


async def test_live_mode_runs_supplementary_sources(
    recorder: _RecordingBind, caplog: pytest.LogCaptureFixture
):
    """In LIVE mode the supplementary HTTP sources DO run (WR-02 only gates backfill)."""
    cycle = datetime(2026, 6, 12, 0, tzinfo=timezone.utc)
    summary = await _ingest_all(
        recorder, "NYC", cycle, mode="live", lead=0
    )

    # GRIB still degrades on the failing hrrr; the others land.
    assert summary["hrrr"] == 0
    assert summary["nbm"] == 1
    assert summary["gfs"] == 1
    assert summary["gefs"] == 31

    # WR-02: live mode runs the live-only sources — nws (1) + openmeteo (2); wethr skips (no key).
    assert summary["nws"] == 1
    assert summary["openmeteo"] == 2
    assert summary["wethr"] == 0


async def test_backfill_skips_live_only_sources_with_structured_log(
    recorder: _RecordingBind, caplog: pytest.LogCaptureFixture
):
    """WR-02: backfill emits a structured live-only skip for nws/openmeteo/wethr (D-11)."""
    cycle = datetime(2026, 6, 12, 0, tzinfo=timezone.utc)
    with caplog.at_level(logging.INFO):
        await _ingest_all(
            recorder, "NYC", cycle, mode="backfill", lead=0
        )

    skip_logs = [r.message for r in caplog.records if "live-only" in r.message]
    for source in ("nws", "openmeteo", "wethr"):
        assert any(f"source={source}" in m for m in skip_logs), (source, skip_logs)
    # No supplementary-source row was recorded (absence = absence).
    assert all(not str(row["model"]).startswith(("nws", "openmeteo", "wethr")) for row in recorder.rows)


async def test_backfill_mode_stamps_publish_latency_not_now(
    recorder: _RecordingBind,
):
    """Backfill rows carry available_at = cycle + PUBLISH_LATENCY, never now() (D-09)."""
    from weatherquant.ingest.available_at import PUBLISH_LATENCY

    cycle = datetime(2026, 6, 12, 0, tzinfo=timezone.utc)
    await _ingest_all(recorder, "NYC", cycle, mode="backfill", lead=0)

    nbm_rows = [r for r in recorder.rows if r["model"] == "nbm"]
    assert nbm_rows, "expected an nbm row"
    # Deterministic backfill stamp — proves no datetime.now leaked into the backfill path.
    assert nbm_rows[0]["available_at"] == cycle + PUBLISH_LATENCY["nbm"]
