"""RED stub — ING-02 / D-04: lead-0 forecast within 3 degC of contemporaneous ASOS.

Turned GREEN by 02-03's ``weatherquant.ingest.grib.lead0_sanity_check``. A breach beyond
3 degC must raise loudly (Pitfall 4) — a silent grid/units bug otherwise corrupts every
downstream calibration. RED at import until 02-03.
"""

from __future__ import annotations

import pytest


def test_lead0_within_3c_of_asos_else_raises():
    # RED: weatherquant.ingest.grib lands in 02-03 (ImportError until then).
    from weatherquant.ingest.grib import lead0_sanity_check

    # A forecast 10 degC off the ASOS observation must raise (D-04 loud breach).
    with pytest.raises(Exception):
        lead0_sanity_check(forecast_k=283.15, asos_k=293.15, tolerance_c=3.0)
