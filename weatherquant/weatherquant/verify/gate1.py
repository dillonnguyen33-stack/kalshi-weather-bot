"""Pre-registered Gate-1 pass/fail logic (VER-07 / D-08): conjunctive, direction-aware verdict.

D-08 (verify subtree-local; anti-p-hacking): the Gate-1 verdict is PRE-REGISTERED. The five pooled
metric CIs must all clear zero in the correct direction — and the metric key-set + criterion are
loaded from a frozen pre-registration file (``load_preregistration``) that is checked against the
live run so the bar can never be moved after seeing the results.

Direction matters: Brier / CRPS / ECE are LOWER-is-better (an edge means the CI of
``WQ − v3`` lies strictly BELOW zero, i.e. ``ci_hi < 0``); ROI / CLV are HIGHER-is-better (the CI
lies strictly ABOVE zero, ``ci_lo > 0``). ``gate1_passes`` is conjunctive — ALL five must pass.

Pure stdlib + the CI tuples — no scipy/sklearn (AST-guarded). Bodies land Wave 3;
``tests/test_gate1.py`` pins the contracts (RED).
"""

from __future__ import annotations

__all__ = ["LOWER_IS_BETTER", "HIGHER_IS_BETTER", "metric_passes", "gate1_passes", "load_preregistration"]

# Metric direction registry (D-08). A lower-is-better metric passes only when the paired
# (WQ − v3) CI lies strictly below zero; a higher-is-better metric only when strictly above.
LOWER_IS_BETTER = {"brier", "crps", "ece"}
HIGHER_IS_BETTER = {"roi", "clv"}


def metric_passes(name: str, ci_lo: float, ci_hi: float) -> bool:
    """Direction-aware single-metric pass test (VER-07).

    For ``name`` in :data:`LOWER_IS_BETTER`, passes iff ``ci_hi < 0`` (the whole paired CI is
    below zero); for ``name`` in :data:`HIGHER_IS_BETTER`, passes iff ``ci_lo > 0``. An unknown
    metric name fails loud. Body lands Wave 3.
    """
    raise NotImplementedError("verify.gate1.metric_passes lands in Wave 3 (VER-07).")


def gate1_passes(cis: dict[str, tuple[float, float]]) -> bool:
    """Conjunctive pre-registered Gate-1 verdict over the five pooled metric CIs (VER-07).

    Asserts the exact pre-registered metric key-set is present, then requires EVERY metric to
    pass :func:`metric_passes` (conjunctive — a single failing metric fails the gate). Body lands
    Wave 3.
    """
    raise NotImplementedError("verify.gate1.gate1_passes lands in Wave 3 (VER-07).")


def load_preregistration(path) -> dict:
    """Load the frozen Gate-1 pre-registration (metric set + criterion) and check it (D-08).

    Reads the pre-registered metric key-set / direction / criterion from ``path`` and fails LOUD
    on any mismatch against the live run's metric set — so the Gate-1 bar can never be moved after
    seeing results (anti-p-hacking). Body lands Wave 3.
    """
    raise NotImplementedError("verify.gate1.load_preregistration lands in Wave 3 (VER-07).")
