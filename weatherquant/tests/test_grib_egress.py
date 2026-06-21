"""GRIB egress guard — the :TMP:2 m subset must stay ~1 MB, never the ~700 MB full file.

Phase-4 fix #6. A Herbie default/search regression that matched the whole .idx inventory would
silently pull the full file. ``_assert_subset_inventory`` refuses an empty or implausibly large
match before the byte-range download. Pure (no Herbie / network): it takes the matched-record
container and checks its length.
"""

from __future__ import annotations

import pytest

from weatherquant.ingest.errors import SanityError
from weatherquant.ingest.grib import _MAX_T2M_RECORDS, _assert_subset_inventory


def test_single_record_match_is_accepted():
    assert _assert_subset_inventory([object()], "hrrr") == 1  # exactly one :TMP:2 m record


def test_whole_file_match_is_refused():
    too_many = [object()] * (_MAX_T2M_RECORDS + 1)
    with pytest.raises(SanityError, match="egress guard"):
        _assert_subset_inventory(too_many, "gfs")


def test_empty_match_is_refused():
    with pytest.raises(SanityError, match="egress guard"):
        _assert_subset_inventory([], "nbm")
