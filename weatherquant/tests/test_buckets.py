"""Bucket-mapping tests (PRC-02, D-04/D-05/D-06) — RED until Wave 1 (04-03).

Covers the two bucket behaviors from the VALIDATION map:

* ``-k parse`` — ``parse_ticker`` round-trips a known Kalshi range label/ticker to its
  inclusive integer-degree ``(lo, hi)`` edges.
* ``-k sum`` — a full integer ladder's bucket probabilities (incl. open tails) sum to ~1.

All ``xfail`` (the stubs raise ``NotImplementedError``) so Wave 1 flips them GREEN without
renaming — the test names match the ``-k`` selectors in 04-VALIDATION.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from weatherquant.price.buckets import bucket_prob, parse_ticker


@pytest.mark.xfail(reason="Wave 1 (04-03) implements parse_ticker", strict=False)
def test_buckets_parse_ticker_round_trip():
    # A "62 to 63" range parses to its inclusive integer-degree bounds.
    lo, hi = parse_ticker("KXHIGHNY-62-63")
    assert (lo, hi) == (62, 63)


@pytest.mark.xfail(reason="Wave 1 (04-03) implements bucket_prob", strict=False)
def test_buckets_ladder_sum_to_one():
    # A full integer ladder around the mean, with open tails, tiles the line ⇒ sum ≈ 1.
    mu, sigma = 70.0, 4.0
    ladder = [(k - 0.5, k + 0.5) for k in range(55, 86)]
    total = 0.0
    # Open lower tail (≤ 54) and open upper tail (≥ 86).
    total += bucket_prob(mu, sigma, -np.inf, 54.5, open_lo=True)
    for lo, hi in ladder:
        total += bucket_prob(mu, sigma, lo, hi)
    total += bucket_prob(mu, sigma, 85.5, np.inf, open_hi=True)
    assert total == pytest.approx(1.0, abs=1e-6)
