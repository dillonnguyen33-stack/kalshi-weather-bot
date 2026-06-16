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
