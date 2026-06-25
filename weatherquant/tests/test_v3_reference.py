"""RED contract for the legacy v3 probability adapter (VER-04 / D-02 / D-03).

Asserts ``weatherquant.verify.v3_reference`` reproduces EVERY frozen golden tuple in
``tests/fixtures/v3_golden.py`` to the legacy rounding exactly — so the v3 baseline in the paired
backtest is the true legacy number (T-06-01). Also asserts the D-03 leak guard: the adapter takes
NO same-day-obs argument (no intraday ASOS override path leaks into the apples-to-apples compare).

Imports are deferred into the test bodies so collection is green while the Wave-2 bodies are RED.
"""

from __future__ import annotations

import inspect

import pytest

from tests.fixtures import v3_golden


@pytest.mark.parametrize(("x", "mean", "spread", "expected"), v3_golden.V3_NORMAL_CDF_GOLDEN)
def test_v3_normal_cdf_matches_golden(x, mean, spread, expected):
    """The adapter CDF matches the verbatim legacy ``_normal_cdf`` to its 6-dp rounding."""
    from weatherquant.verify import v3_reference

    assert v3_reference.v3_normal_cdf(x, mean, spread) == pytest.approx(expected, abs=0.0)


@pytest.mark.parametrize(("cm", "spread", "lo", "hi", "expected"), v3_golden.V3_BUCKET_PROB_GOLDEN)
def test_v3_bucket_prob_matches_golden(cm, spread, lo, hi, expected):
    """The adapter bucket prob matches the verbatim legacy ensemble branch (spread floor, clamp)."""
    from weatherquant.verify import v3_reference

    assert v3_reference.v3_bucket_prob(cm, spread, lo, hi) == pytest.approx(expected, abs=0.0)


def test_v3_adapter_reads_no_same_day_obs_d03_leak_guard():
    """D-03 exclusion: no v3_reference signature accepts an obs/observation argument (no ASOS leak)."""
    from weatherquant.verify import v3_reference

    for name in ("v3_normal_cdf", "v3_bucket_prob", "v3_bucket_probs"):
        params = set(inspect.signature(getattr(v3_reference, name)).parameters)
        leaked = {p for p in params if "obs" in p.lower()}
        assert not leaked, (
            f"{name} exposes a same-day-obs parameter {leaked} — the v3 adapter must be the pure "
            f"ENSEMBLE math only (D-03); no intraday ASOS override in the Gate-1 comparison."
        )
