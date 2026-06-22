"""Scheduler per-city graceful degradation — one bad city must not abort the others (D-11).

``orchestrator.ingest_cycle`` deliberately re-raises a ``CorrectnessError`` (e.g. a
``TargetDateError`` from an impossible settlement window). The orchestrator's own fan-out stays
bare so that alarm propagates, but the SCHEDULER is the wrong layer to inherit it: a single bad
window must not kill every other city for the model. The job bodies gather with
``return_exceptions=True``, log each failure, and log a partial-success summary.

Offline: ``get_engine`` and ``ingest_cycle`` are stubbed; no DB, no network.
"""

from __future__ import annotations

import logging

import pytest

from weatherquant import scheduler
from weatherquant.ingest.errors import TargetDateError
from weatherquant.registry import CITIES


@pytest.fixture
def stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler, "get_engine", lambda: object())


async def test_grib_job_one_city_alarm_does_not_abort_others(
    stub_engine, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """A TargetDateError on one city still lets the other cities run; the job logs partial success."""
    failed_city = sorted(CITIES)[0]
    seen: list[str] = []

    async def _fake_ingest_cycle(_bind, _model, city, _cycle, *, mode, lead):  # noqa: ANN001
        seen.append(city)
        if city == failed_city:
            raise TargetDateError("no settlement window contains valid instant (D-16)")
        return 1

    monkeypatch.setattr(scheduler.orchestrator, "ingest_cycle", _fake_ingest_cycle)

    with caplog.at_level(logging.INFO):
        await scheduler._ingest_grib_all_cities("hrrr", 1)

    # Every city was attempted — the failing one did NOT abort the rest.
    assert set(seen) == set(CITIES)
    # The alarm stayed visible AND the job logged a partial-success summary.
    assert any(
        f"city={failed_city}" in r.message and "failed" in r.message.lower()
        for r in caplog.records
    )
    assert any(
        "ok=%d" % (len(CITIES) - 1) in r.message and "failed=1" in r.message
        for r in caplog.records
    )


async def test_obs_job_one_city_failure_does_not_abort_others(
    stub_engine, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    """The obs/AFD job degrades per city the same way (obs raising for one city)."""
    failed_city = sorted(CITIES)[0]
    obs_seen: list[str] = []

    async def _fake_ingest_obs(_bind, city, _date, **_kw):  # noqa: ANN001
        obs_seen.append(city)
        if city == failed_city:
            raise TargetDateError("boom")
        return 1

    async def _fake_ingest_afd(_bind, _city, _date, **_kw):  # noqa: ANN001
        return 1

    monkeypatch.setattr(scheduler.orchestrator, "ingest_obs", _fake_ingest_obs)
    monkeypatch.setattr(scheduler.orchestrator, "ingest_afd", _fake_ingest_afd)

    with caplog.at_level(logging.INFO):
        await scheduler._ingest_obs_all_cities()

    assert set(obs_seen) == set(CITIES)
    assert any("failed=1" in r.message for r in caplog.records)
