"""GRIB egress guard — the :TMP:2 m subset must stay ~1 MB, never the ~700 MB full file.

Phase-4 fix #6. A Herbie default/search regression that matched the whole .idx inventory would
silently pull the full file. ``_assert_subset_inventory`` refuses an empty or implausibly large
match before the byte-range download. Pure (no Herbie / network): it takes the matched-record
container and checks its length.
"""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime

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


def test_probe_failure_never_takes_an_unguarded_full_download(monkeypatch):
    """When the .idx probe fails, fetch_t2m must call download(errors='raise'), never a full fetch.

    Herbie's DEFAULT download(errors='warn') silently pulls the full ~700 MB file when the .idx is
    missing. fetch_t2m must pass errors='raise' so a probe failure can only ever fail loud — the
    egress guard the to-do flags as bypassable. A fake ``herbie`` module captures the kwarg and
    mimics Herbie's fail-loud behavior; the unguarded full-file return must never be reached.
    """
    calls: dict[str, object] = {}

    class _FakeHerbie:
        def __init__(self, _date, **_kw):
            pass

        def inventory(self, _search):
            raise RuntimeError("transient .idx probe failure")

        def download(self, _search, *, errors="warn"):
            calls["errors"] = errors
            if errors == "raise":
                # Mimic Herbie: a subset request with a missing .idx fails loud.
                raise ValueError("Index file not found; cannot download subset")
            calls["full_file_downloaded"] = True  # the ~700 MB path — must NOT happen
            return "/tmp/full_file.grib2"

    fake_mod = types.ModuleType("herbie")
    fake_mod.Herbie = _FakeHerbie  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "herbie", fake_mod)

    from weatherquant.ingest import grib

    with pytest.raises(ValueError, match="cannot download subset"):
        grib.fetch_t2m("hrrr", datetime(2026, 6, 12, tzinfo=UTC), 0)

    assert calls["errors"] == "raise"  # the fail-loud kwarg, not the warn full-fetch default
    assert "full_file_downloaded" not in calls  # the unguarded full path was never taken
