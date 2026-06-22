"""ING-02 / D-04: lead-0 forecast within 3 degC of contemporaneous ASOS.

GREEN by 02-02's ``weatherquant.ingest.grib.lead0_sanity_check``. A breach beyond the
tolerance must raise loudly (Pitfall 4) — a silent grid/units bug otherwise corrupts every
downstream calibration. DEN (1656 m) is authorized a 4 degC band per RESEARCH Pitfall 4.
"""

from __future__ import annotations

import pytest

from weatherquant.ingest.grib import lead0_sanity_check


def test_lead0_within_3c_of_asos_else_raises():
    # A forecast 10 degC off the ASOS observation must raise (D-04 loud breach).
    with pytest.raises(Exception):
        lead0_sanity_check(forecast_k=283.15, asos_k=293.15, tolerance_c=3.0)


def test_lead0_within_tolerance_passes():
    # 2 degC apart, under the 3 degC default -> no raise (returns None).
    assert lead0_sanity_check(forecast_k=295.15, asos_k=293.15) is None


def test_lead0_default_tolerance_is_3c():
    # Exactly 3 degC passes; just over 3 degC breaches under the default.
    assert lead0_sanity_check(forecast_k=296.15, asos_k=293.15) is None
    with pytest.raises(ValueError):
        lead0_sanity_check(forecast_k=296.65, asos_k=293.15)  # 3.5 degC


def test_den_relaxes_to_4c():
    # DEN: 3.5 degC apart passes under the 4 degC band, but 4.5 degC still breaches.
    assert lead0_sanity_check(forecast_k=296.65, asos_k=293.15, city_code="DEN") is None
    with pytest.raises(ValueError):
        lead0_sanity_check(forecast_k=297.65, asos_k=293.15, city_code="DEN")  # 4.5 degC


def test_non_den_city_keeps_3c_band():
    # A non-DEN city does NOT get the wider band: 3.5 degC breaches.
    with pytest.raises(ValueError):
        lead0_sanity_check(forecast_k=296.65, asos_k=293.15, city_code="NYC")  # 3.5 degC


# --- 1.2: the lead-0 probe must be WIRED into the live ingest path, not just exported. -------
# These prove the orchestrator calls lead0_sanity_check at lead 0 against the contemporaneous
# ASOS read. On the OLD code (no production caller) the breach below would have stored a row
# from a wrong snap/unit/grid and silently corrupted every downstream fit.

from datetime import datetime, timezone  # noqa: E402

from weatherquant.ingest import obs, orchestrator  # noqa: E402
from weatherquant.ingest.errors import SanityError  # noqa: E402

_CYCLE = datetime(2026, 6, 12, 0, tzinfo=timezone.utc)


def _patch_grib(monkeypatch, forecast_k: float) -> None:
    monkeypatch.setattr(orchestrator.grib, "fetch_t2m", lambda *a, **k: object())
    monkeypatch.setattr(
        orchestrator.grib,
        "snap_city",
        lambda *a, **k: (forecast_k, 40.779, -73.969, 100.0),
    )


async def test_orchestrator_lead0_breach_raises(monkeypatch):
    """A lead-0 forecast 10 degC off the contemporaneous ASOS raises through ingest_cycle."""
    _patch_grib(monkeypatch, forecast_k=293.15)

    async def _asos(*a, **k):
        return 283.15  # 10 degC below the forecast -> breach

    monkeypatch.setattr(obs, "asos_lead0_kelvin", _asos)
    with pytest.raises(SanityError):
        await orchestrator.ingest_cycle(object(), "hrrr", "NYC", _CYCLE, mode="live", lead=0)


async def test_orchestrator_lead0_within_tolerance_ingests(monkeypatch):
    """A lead-0 forecast within tolerance ingests normally (the row is written)."""
    rows: list[dict] = []
    _patch_grib(monkeypatch, forecast_k=294.15)
    monkeypatch.setattr(
        orchestrator, "insert_forecast", lambda _b, **kw: (rows.append(kw) or 1)
    )

    async def _asos(*a, **k):
        return 293.15  # 1 degC from the forecast -> within the 3 degC band

    monkeypatch.setattr(obs, "asos_lead0_kelvin", _asos)
    n = await orchestrator.ingest_cycle(object(), "hrrr", "NYC", _CYCLE, mode="live", lead=0)
    assert n == 1
    assert len(rows) == 1


async def test_orchestrator_lead0_skips_probe_when_asos_absent(monkeypatch):
    """No contemporaneous ASOS -> the probe is skipped (can't verify), the row still ingests."""
    rows: list[dict] = []
    _patch_grib(monkeypatch, forecast_k=350.0)  # would breach IF a probe ran
    monkeypatch.setattr(
        orchestrator, "insert_forecast", lambda _b, **kw: (rows.append(kw) or 1)
    )

    async def _no_asos(*a, **k):
        return None

    monkeypatch.setattr(obs, "asos_lead0_kelvin", _no_asos)
    n = await orchestrator.ingest_cycle(object(), "hrrr", "NYC", _CYCLE, mode="live", lead=0)
    assert n == 1
