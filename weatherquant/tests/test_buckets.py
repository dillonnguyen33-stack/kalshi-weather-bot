"""Bucket-mapping tests (PRC-02, D-04/D-05/D-06).

Covers the two bucket behaviors from the VALIDATION map:

* ``-k parse`` — ``parse_ticker`` round-trips a known Kalshi range label/ticker to its
  ``(lo, hi, open_lo, open_hi)`` edges, prefers structured strikes, and fails loud on
  malformed input.
* ``-k sum`` — a full integer ladder's bucket probabilities (incl. open tails) sum to ~1,
  via CDF differencing over the centralized ``_HALF`` half-degree offset.
* ``-k real_ladder`` — a representative ``KXHIGH``-shaped integer ladder built end-to-end
  through the public seam (``parse_ticker`` structured strikes → ``integers_in_bucket`` →
  ``bucket_probs``) tiles to ~1 AND keeps the modal bucket on the blended μ (Pitfall 1's two
  warning signs absent). The LIVE ``KXHIGH`` cross-check is deferred to Phase 5 (no Kalshi API
  this session, user decision 2026-06-17); this guards the principled mapping in the meantime.

Flipped GREEN by Wave 1 (04-03); the test names keep the ``-k`` selectors in
04-VALIDATION.md so no renaming is needed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from weatherquant.price.buckets import (
    bucket_prob,
    bucket_probs,
    integers_in_bucket,
)
from weatherquant.price.ticker import parse_ticker

# ---------------------------------------------------------------------------
# -k sum : CDF differencing + centralized _HALF + ladder-sum property
# ---------------------------------------------------------------------------


def test_buckets_integers_in_bucket_half_degree_span():
    # The closed "62 to 63" label covers integers {62, 63} → continuous [61.5, 64.0).
    c_lo, c_hi = integers_in_bucket(62, 63)
    assert c_lo == pytest.approx(61.5)
    assert c_hi == pytest.approx(63.5)


def test_buckets_integers_in_bucket_open_tails_use_sentinels():
    c_lo, c_hi = integers_in_bucket(None, 54, open_lo=True)
    assert c_lo == -math.inf
    assert c_hi == pytest.approx(54.5)
    c_lo, c_hi = integers_in_bucket(86, None, open_hi=True)
    assert c_lo == pytest.approx(85.5)
    assert c_hi == math.inf


def test_buckets_bucket_prob_closed_equals_cdf_difference():
    mu, sigma = 70.0, 4.0
    c_lo, c_hi = integers_in_bucket(70, 70)  # single degree → [69.5, 70.5)
    from weatherquant.calibrate.crps import normal_cdf

    expected = float(normal_cdf(np.array([(c_hi - mu) / sigma]))[0]) - float(
        normal_cdf(np.array([(c_lo - mu) / sigma]))[0]
    )
    assert bucket_prob(mu, sigma, c_lo, c_hi) == pytest.approx(expected)


def test_buckets_ladder_sum_to_one():
    # A full integer ladder around the mean, with open tails, tiles the line ⇒ sum ≈ 1.
    mu, sigma = 70.0, 4.0
    ladder: list[tuple[float, float, bool, bool]] = []
    ladder.append((-math.inf, 54.5, True, False))  # open lower tail (≤ 54)
    for k in range(55, 86):
        c_lo, c_hi = integers_in_bucket(k, k)
        ladder.append((c_lo, c_hi, False, False))
    ladder.append((85.5, math.inf, False, True))  # open upper tail (≥ 86)
    probs = bucket_probs(mu, sigma, ladder)
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)


def test_buckets_real_ladder_sums_to_one_and_modal_bucket_on_mu():
    # A representative KXHIGH-shaped integer ladder (single-degree buckets between two open
    # tails), built end-to-end through the PUBLIC seam — parse_ticker (structured Kalshi
    # strikes) → integers_in_bucket → bucket_probs — under the deferred (Phase-5-locked)
    # [k − _HALF, k + _HALF) coverage. The live KXHIGH cross-check is deferred to Phase 5
    # (no Kalshi API this session); this asserts the two Pitfall-1 warning signs are absent:
    #   1. the full ladder tiles the line ⇒ probabilities sum to ~1 (no gap/overlap), and
    #   2. the modal bucket sits ON the blended μ, not one degree off (silent ±1° mass bias).
    mu, sigma = 70.0, 4.0
    lo_deg, hi_deg = 55, 85  # closed single-degree buckets KXHIGH{55..85}, open tails outside

    ladder: list[tuple[float, float, bool, bool]] = []
    bucket_centers: list[float | None] = []

    # Open lower tail: "≤ (lo_deg − 1)" via the structured 'less' strike.
    lo, hi, open_lo, open_hi = parse_ticker(cap_strike=lo_deg - 1, strike_type="less")
    ladder.append((*integers_in_bucket(lo, hi, open_lo, open_hi), open_lo, open_hi))
    bucket_centers.append(None)  # tail has no finite center

    # Closed single-degree buckets, each built from the structured between-strikes the live
    # Kalshi market record supplies (floor_strike == cap_strike == k for a 1°F bucket).
    for k in range(lo_deg, hi_deg + 1):
        lo, hi, open_lo, open_hi = parse_ticker(
            floor_strike=k, cap_strike=k, strike_type="between"
        )
        assert (lo, hi, open_lo, open_hi) == (k, k, False, False)
        ladder.append((*integers_in_bucket(lo, hi, open_lo, open_hi), open_lo, open_hi))
        bucket_centers.append(float(k))

    # Open upper tail: "≥ (hi_deg + 1)" via the structured 'greater' strike.
    lo, hi, open_lo, open_hi = parse_ticker(floor_strike=hi_deg + 1, strike_type="greater")
    ladder.append((*integers_in_bucket(lo, hi, open_lo, open_hi), open_lo, open_hi))
    bucket_centers.append(None)

    probs = bucket_probs(mu, sigma, ladder)

    # 1. Sum-to-one within 1e-9 (no gap/overlap across the real-shaped ladder).
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)

    # 2. Modal bucket sits exactly on the blended μ (no one-degree shift). With μ an integer
    #    and symmetric Gaussian mass, the densest closed bucket must be centered on μ itself.
    modal_idx = int(np.argmax(probs))
    assert bucket_centers[modal_idx] == pytest.approx(mu)


def test_buckets_bucket_prob_open_tails_use_one_sided_limits():
    mu, sigma = 70.0, 4.0
    # Open-low tail uses 0.0 at the bottom; open-high tail uses 1.0 at the top.
    p_low = bucket_prob(mu, sigma, -math.inf, 54.5, open_lo=True)
    p_high = bucket_prob(mu, sigma, 85.5, math.inf, open_hi=True)
    assert 0.0 < p_low < 1.0
    assert 0.0 < p_high < 1.0


def test_buckets_bucket_prob_fails_loud_on_bad_sigma():
    with pytest.raises(ValueError):
        bucket_prob(70.0, 0.0, 69.5, 70.5)
    with pytest.raises(ValueError):
        bucket_prob(70.0, -1.0, 69.5, 70.5)


def test_buckets_bucket_prob_fails_loud_on_non_finite():
    with pytest.raises(ValueError):
        bucket_prob(float("nan"), 4.0, 69.5, 70.5)
    with pytest.raises(ValueError):
        bucket_prob(70.0, float("inf"), 69.5, 70.5)


def test_buckets_bucket_prob_fails_loud_on_both_open_ends():
    # 4.1: a bucket open on BOTH ends would silently return 1.0; it must raise instead.
    with pytest.raises(ValueError):
        bucket_prob(70.0, 4.0, -math.inf, math.inf, open_lo=True, open_hi=True)


# ---------------------------------------------------------------------------
# -k parse : pure ticker/strike → (lo, hi, open_lo, open_hi), fail-loud
# ---------------------------------------------------------------------------


def test_buckets_parse_ticker_round_trip():
    # A "62 to 63" range ticker parses to its inclusive integer-degree bounds, both closed.
    lo, hi, open_lo, open_hi = parse_ticker("KXHIGHNY-62-63")
    assert (lo, hi) == (62, 63)
    assert not open_lo and not open_hi


def test_buckets_parse_label_closed_range():
    lo, hi, open_lo, open_hi = parse_ticker(label="62° to 63°")
    assert (lo, hi) == (62, 63)
    assert not open_lo and not open_hi
    # ASCII variant.
    lo, hi, open_lo, open_hi = parse_ticker(label="62 to 63")
    assert (lo, hi) == (62, 63)


def test_buckets_parse_label_open_low():
    lo, hi, open_lo, open_hi = parse_ticker(label="≤ 55°")
    assert hi == 55 and open_lo and not open_hi
    lo, hi, open_lo, open_hi = parse_ticker(label="55° or below")
    assert hi == 55 and open_lo and not open_hi


def test_buckets_parse_label_open_high():
    lo, hi, open_lo, open_hi = parse_ticker(label="≥ 80°")
    assert lo == 80 and open_hi and not open_lo
    lo, hi, open_lo, open_hi = parse_ticker(label="80° or above")
    assert lo == 80 and open_hi and not open_lo


def test_buckets_parse_prefers_structured_strikes_over_label():
    # Structured strikes win even when a (deliberately wrong) label is also supplied.
    lo, hi, open_lo, open_hi = parse_ticker(
        floor_strike=62,
        cap_strike=63,
        strike_type="between",
        label="99° to 100°",
    )
    assert (lo, hi) == (62, 63)
    assert not open_lo and not open_hi


def test_buckets_parse_structured_open_tails():
    lo, hi, open_lo, open_hi = parse_ticker(cap_strike=55, strike_type="less")
    assert hi == 55 and open_lo
    lo, hi, open_lo, open_hi = parse_ticker(floor_strike=80, strike_type="greater")
    assert lo == 80 and open_hi


@pytest.mark.parametrize(
    "kwargs",
    [
        {"label": ""},
        {"label": "   "},
        {"label": "not a range"},
        {"ticker": "KXHIGHNY-63-62"},  # inverted lo > hi
        {"label": "63 to 62"},  # inverted lo > hi
        {},  # nothing supplied
    ],
)
def test_buckets_parse_fails_loud_on_malformed(kwargs):
    with pytest.raises(ValueError):
        parse_ticker(**kwargs)
