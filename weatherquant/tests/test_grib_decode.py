"""RED stub — ING-01/02: decode a vendored GRIB2 subset to t2m in Kelvin (D-01/D-02/D-03).

Turned GREEN by 02-03's ``weatherquant.ingest.grib``. Until then this fails at import
(the module does not exist yet) — the Wave-0 RED predecessor for the GRIB decode path.
The fixture itself already decodes offline (proven in 02-01); what is missing is the
``decode_t2m`` edge function that asserts units=="K" and returns a plain ``np.ndarray``.
"""

from __future__ import annotations


def test_decode_vendored_grib_returns_kelvin_ndarray(grib_fixture):
    # RED: weatherquant.ingest.grib lands in 02-03 (ImportError until then).
    from weatherquant.ingest.grib import decode_t2m

    path = grib_fixture("hrrr")
    field = decode_t2m(path)
    # Decoded 2-m temperature must be in Kelvin (D-03) and a plain ndarray (D-02).
    assert field.units == "K"
    assert field.values.ndim == 2
