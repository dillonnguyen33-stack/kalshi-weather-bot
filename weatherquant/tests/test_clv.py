"""RED stub — closing-line value (PAP-04, 05-03 fills this GREEN).

CLV measures a fill against the volume-weighted CLOSING mid over the final window before the
market settles. Sign: a yes BUY filled BELOW the closing mid is POSITIVE CLV (we bought
cheap), above is NEGATIVE. The closing mid is golden over ``closing_window_snapshots``
(volume-weighted = 51.4¢). The ``-k window`` selector matches the VALIDATION command ``pytest
tests/test_clv.py -k window``: the closing window MUST be anchored on
``time.settlement_window(city, day).end_utc`` (the half-open EXCLUSIVE end) — NEVER a
hand-rolled civil-time clock (D-10, the v3 founding bug). ``clv.py`` is a PURE module.

Wave-0 RED stub: ``importorskip`` the not-yet-existing ``weatherquant.market.clv``.
"""

from __future__ import annotations

import pytest

clv = pytest.importorskip("weatherquant.market.clv")


@pytest.mark.xfail(reason="RED — 05-03 implements CLV", strict=False)
def test_clv_sign_positive_when_fill_better_than_close(closing_window_snapshots):
    """A yes BUY at 48¢ vs a 51.4¢ closing mid yields POSITIVE CLV."""
    raise NotImplementedError("05-03: clv = (close_mid - fill_price) for a buy")


@pytest.mark.xfail(reason="RED — 05-03 implements CLV", strict=False)
def test_clv_sign_negative_when_fill_worse_than_close(closing_window_snapshots):
    """A yes BUY at 55¢ vs a 51.4¢ closing mid yields NEGATIVE CLV."""
    raise NotImplementedError("05-03: clv sign flips when fill is worse than close")


@pytest.mark.xfail(reason="RED — 05-03 implements CLV", strict=False)
def test_volume_weighted_closing_mid_golden(closing_window_snapshots):
    """The closing mid is the volume-weighted mean (51.4¢) over the window snapshots."""
    raise NotImplementedError("05-03: vol-weighted mid = Σ(mid*vol)/Σvol")


@pytest.mark.xfail(reason="RED — 05-03 anchors the window on settlement_window", strict=False)
def test_closing_window_anchored_on_settlement_window_end():
    """The closing window anchors on ``time.settlement_window(...).end_utc``, not a re-derived clock."""
    raise NotImplementedError("05-03: reuse time.settlement_window, never a hand-rolled clock")
