"""RED contract for the Gate-1 metric core (VER-01).

Asserts the eventual GREEN behavior of ``weatherquant.verify.metrics`` against synthetic Gaussian
fixtures. The imports are DEFERRED into each test body (mirroring ``test_cli_parity.py``'s "RED
until then" pattern) so collection is green while the Wave-2 implementation is RED (every call hits
``NotImplementedError`` until the bodies land).
"""

from __future__ import annotations

import numpy as np
import pytest


def _calibrated_binary(seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """A perfectly-calibrated synthetic binary forecast: f ~ U(0,1), o ~ Bernoulli(f)."""
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.0, 1.0, n)
    o = (rng.uniform(0.0, 1.0, n) < f).astype(np.float64)
    return f, o


def test_crps_blend_matches_calibrate_crps_norm_for_a_single_gaussian():
    """crps_blend reduces to the mean of calibrate.crps.crps_norm for one Gaussian (D-04 reuse)."""
    from weatherquant.calibrate.crps import crps_norm
    from weatherquant.verify import metrics

    rng = np.random.default_rng(7)
    mu = rng.normal(70.0, 5.0, 500)
    sigma = np.full(500, 2.5)
    y = rng.normal(mu, sigma)
    expected = float(crps_norm(mu, sigma, y).mean())
    assert metrics.crps_blend(mu, sigma, y) == pytest.approx(expected, rel=1e-9)


def test_ece_equal_count_zero_for_calibrated_positive_for_biased():
    """Equal-count ECE is ~0 for a calibrated forecast and clearly > 0 for a biased one."""
    from weatherquant.verify import metrics

    f, o = _calibrated_binary(seed=2, n=20_000)
    ece_calibrated = metrics.ece_equal_count(f, o, n_bins=10)
    # A systematic +0.2 over-forecast (clipped) is demonstrably miscalibrated.
    f_biased = np.clip(f + 0.2, 0.0, 1.0)
    ece_biased = metrics.ece_equal_count(f_biased, o, n_bins=10)
    assert ece_calibrated < 0.03
    assert ece_biased > ece_calibrated + 0.05


# --- 06-09 Task 1: side-aware roi_from_fills (mirror clv_cents YES/NO sign orientation) ----------


def _roi_fill(avg_price_cents: float, count: float = 1.0, fee: float = 0.0) -> dict:
    """A minimal fills row carrying the float ``detail['avg_price_cents']`` (never the rounded price)."""
    return {"count": count, "fee": fee, "detail": {"avg_price_cents": avg_price_cents}}


def test_roi_from_fills_yes_buy_settles_yes_is_unchanged_baseline():
    """A YES buy of 1 @ 40c that settles YES: payoff 100c, net 60c, entry 40c → ROI 1.5 (baseline)."""
    from weatherquant.verify import metrics

    roi = metrics.roi_from_fills([_roi_fill(40.0)], [True], ["yes"])
    assert roi == pytest.approx(60.0 / 40.0)


def test_roi_from_fills_yes_buy_settles_no_is_a_loss():
    """A YES buy of 1 @ 40c that settles NO: payoff 0, net -40c → ROI -1.0."""
    from weatherquant.verify import metrics

    roi = metrics.roi_from_fills([_roi_fill(40.0)], [False], ["yes"])
    assert roi == pytest.approx(-1.0)


def test_roi_from_fills_no_buy_settles_no_is_a_win():
    """The defect regression: a side='no' fill @ 40c that settles NO is a WIN.

    The NO mirror pays ``100 - 40 = 60`` net per the clv_cents sign orientation (a NO position wins
    when the bucket settles NO) — it must NOT be scored as a loss. ROI = 60/40 = 1.5 > 0.
    """
    from weatherquant.verify import metrics

    roi = metrics.roi_from_fills([_roi_fill(40.0)], [False], ["no"])
    assert roi > 0
    assert roi == pytest.approx(60.0 / 40.0)


def test_roi_from_fills_no_buy_settles_yes_is_a_loss():
    """A side='no' fill @ 40c that settles YES is a LOSS (payoff 0, net -40c → ROI < 0)."""
    from weatherquant.verify import metrics

    roi = metrics.roi_from_fills([_roi_fill(40.0)], [True], ["no"])
    assert roi < 0
    assert roi == pytest.approx(-1.0)


def test_roi_from_fills_sell_alias_settles_no_is_a_win():
    """``side='sell'`` is the NO-equivalent alias (mirrors _settle_window_fills normalization)."""
    from weatherquant.verify import metrics

    roi = metrics.roi_from_fills([_roi_fill(40.0)], [False], ["sell"])
    assert roi == pytest.approx(60.0 / 40.0)


def test_roi_from_fills_length_mismatch_raises():
    """fills / settled_yes / sides length mismatch fails loud (D-01 — never silently truncate)."""
    from weatherquant.verify import metrics

    with pytest.raises(ValueError):
        metrics.roi_from_fills([_roi_fill(40.0)], [True, False], ["yes", "yes"])
    with pytest.raises(ValueError):
        metrics.roi_from_fills([_roi_fill(40.0)], [True], ["yes", "no"])


def test_roi_from_fills_price_out_of_cents_range_raises():
    """The ``[0, 100]`` cents guard is preserved (a price > 100 is a unit bug → raise)."""
    from weatherquant.verify import metrics

    with pytest.raises(ValueError):
        metrics.roi_from_fills([_roi_fill(140.0)], [True], ["yes"])
