"""RED contract for the pre-registered Gate-1 verdict logic (VER-07 / D-08).

Covers the direction-aware single-metric test, the conjunctive gate, and the anti-p-hacking
pre-registration check:

* ``metric_passes`` encodes the correct direction — Brier/CRPS/ECE pass only when the paired CI is
  strictly BELOW zero (ci_hi < 0); ROI/CLV only when strictly ABOVE zero (ci_lo > 0).
* ``gate1_passes`` is conjunctive (all five) and asserts the exact metric key-set.
* ``load_preregistration`` fails loud on a live-run metric-set mismatch (D-08).

Imports are deferred so collection stays green while the implementation is RED.
"""

from __future__ import annotations

import json

import pytest

_LOWER = ("brier", "crps", "ece")  # lower-is-better → pass iff ci_hi < 0
_HIGHER = ("roi", "clv")  # higher-is-better → pass iff ci_lo > 0


@pytest.mark.parametrize("name", _LOWER)
def test_lower_is_better_metric_direction(name):
    """A lower-is-better metric passes ONLY when the whole paired CI lies below zero."""
    from weatherquant.verify import gate1

    assert gate1.metric_passes(name, -0.05, -0.01) is True  # strictly below zero → edge
    assert gate1.metric_passes(name, -0.05, 0.01) is False  # straddles zero → no edge
    assert gate1.metric_passes(name, 0.01, 0.05) is False  # above zero → worse than v3


@pytest.mark.parametrize("name", _HIGHER)
def test_higher_is_better_metric_direction(name):
    """A higher-is-better metric passes ONLY when the whole paired CI lies above zero."""
    from weatherquant.verify import gate1

    assert gate1.metric_passes(name, 0.01, 0.05) is True  # strictly above zero → edge
    assert gate1.metric_passes(name, -0.01, 0.05) is False  # straddles zero → no edge
    assert gate1.metric_passes(name, -0.05, -0.01) is False  # below zero → worse than v3


@pytest.mark.parametrize("name", _HIGHER)
@pytest.mark.parametrize("x", [0.01, 0.5, 1.0, 42.0])
def test_higher_is_better_rejects_a_degenerate_zero_width_ci(name, x):
    """CR-01 (belt-and-suspenders): a zero-width CI (ci_lo == ci_hi) is NOT a pass, any x > 0.

    The BLOCKER hole: ``metric_passes("roi", x, x)`` with ``x > 0`` formerly returned ``True``
    (``ci_lo > 0`` alone), so a degenerate point CI manufactured a money-gate PASS. A zero-width
    interval cannot HONESTLY exclude zero — it is a single point, not a confidence interval — so
    HIGHER_IS_BETTER must require strictly positive width ``ci_hi > ci_lo`` in addition to
    ``ci_lo > 0``.
    """
    from weatherquant.verify import gate1

    assert gate1.metric_passes(name, x, x) is False  # degenerate point at a profit is NOT a pass


@pytest.mark.parametrize("name", _HIGHER)
def test_higher_is_better_rejects_an_inverted_ci(name):
    """CR-01: an inverted CI (ci_hi < ci_lo) can never pass a HIGHER_IS_BETTER metric."""
    from weatherquant.verify import gate1

    assert gate1.metric_passes(name, 0.05, 0.01) is False  # hi < lo, both > 0 → not a pass


def test_gate1_passes_is_conjunctive_over_the_exact_key_set():
    """All five metrics passing → gate passes; any one failing → gate fails."""
    from weatherquant.verify import gate1

    passing = {
        "brier": (-0.05, -0.01), "crps": (-0.05, -0.01), "ece": (-0.05, -0.01),
        "roi": (0.01, 0.05), "clv": (0.01, 0.05),
    }
    assert gate1.gate1_passes(passing) is True

    one_failing = dict(passing)
    one_failing["ece"] = (-0.01, 0.02)  # straddles zero → fails
    assert gate1.gate1_passes(one_failing) is False


def test_gate1_passes_rejects_a_wrong_metric_key_set():
    """A CI dict missing a pre-registered metric (or carrying an extra one) fails loud (D-08)."""
    from weatherquant.verify import gate1

    missing_clv = {
        "brier": (-0.05, -0.01), "crps": (-0.05, -0.01), "ece": (-0.05, -0.01),
        "roi": (0.01, 0.05),
    }
    with pytest.raises((ValueError, KeyError, AssertionError)):
        gate1.gate1_passes(missing_clv)


def test_load_preregistration_fails_loud_on_metric_set_mismatch(tmp_path):
    """A pre-registration whose metric set disagrees with the live run raises (anti-p-hacking, D-08)."""
    from weatherquant.verify import gate1

    prereg = tmp_path / "gate1_preregistration.json"
    prereg.write_text(json.dumps({"metrics": ["brier", "crps", "ece", "roi"]}))  # missing clv
    loaded = gate1.load_preregistration(prereg)
    # The loaded pre-registration's metric set must NOT silently equal the canonical five.
    assert set(loaded["metrics"]) != set(gate1.LOWER_IS_BETTER | gate1.HIGHER_IS_BETTER)
