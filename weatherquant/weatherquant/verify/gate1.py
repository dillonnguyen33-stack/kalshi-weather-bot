"""Pre-registered Gate-1 pass/fail logic (VER-07 / D-06 / D-07): conjunctive, direction-aware verdict.

D-06 (verify subtree-local; frozen conjunctive PASS rule): Gate-1 PASSES iff for EVERY metric in
``{brier, crps, ece, roi, clv}`` the two-sided 95% paired day-block-bootstrap CI on the pooled
delta (weatherquant − v3) excludes zero in weatherquant's favorable direction. Any single metric
whose CI touches or crosses zero → Gate-1 does NOT pass → no real money. This rule is frozen
(D-13 in CONTEXT) and must not be re-opened after results are seen.

D-07 (verify subtree-local; pre-registration freeze guard / anti-p-hacking): the Gate-1 endpoint
is PRE-REGISTERED to a frozen JSON artifact BEFORE the verdict is computed. ``write_preregistration``
refuses to overwrite an existing pre-registration (the freeze is irreversible);
``load_preregistration`` reads it and ``assert_matches_preregistration`` fails LOUD if the live
run's metrics / seed / primary lead / test window / sign map differ from the frozen spec — so the
bar can never be moved (seed-shopping, lead/window-shifting, or metric-swapping) after seeing the
results (D-08 in CONTEXT).

Direction matters (RESEARCH §Pitfall 7 — a sign flip silently inverts the verdict): Brier / CRPS /
ECE are LOWER-is-better (an edge means the CI of ``WQ − v3`` lies strictly BELOW zero, i.e.
``ci_hi < 0``); ROI / CLV are HIGHER-is-better (the CI lies strictly ABOVE zero, ``ci_lo > 0``).
``gate1_passes`` is conjunctive — ALL five must pass; Holm is NOT applied to these five primaries
(the conjunction is already conservative by construction — see ``verify.bootstrap``).

Pure stdlib (``json``) + the sign-map constants — no scipy/sklearn/matplotlib (AST-guarded).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "LOWER_IS_BETTER",
    "HIGHER_IS_BETTER",
    "GATE1_METRICS",
    "metric_passes",
    "gate1_passes",
    "load_preregistration",
    "write_preregistration",
    "assert_matches_preregistration",
]

# Metric direction registry (D-06). A lower-is-better metric passes only when the paired
# (WQ − v3) CI lies strictly below zero; a higher-is-better metric only when strictly above.
# A sign flip here silently inverts every verdict (RESEARCH §Pitfall 7) — guarded by the
# straddle/below/above unit tests in tests/test_gate1.py.
LOWER_IS_BETTER = {"brier", "crps", "ece"}
HIGHER_IS_BETTER = {"roi", "clv"}

# The exact pre-registered Gate-1 metric key-set — the conjunctive D-06 rule scores ALL five and
# nothing else. ``gate1_passes`` asserts the live CI dict carries exactly this set (fail loud).
GATE1_METRICS = LOWER_IS_BETTER | HIGHER_IS_BETTER

# Pre-registration parameters that the live run must reproduce verbatim (anti-p-hacking, D-07).
# Re-tuning ANY of these after seeing results is the p-hacking vector the freeze guard blocks.
_PREREG_GUARDED_KEYS = ("metrics", "seed", "primary_lead", "test_window", "sign_map")


def metric_passes(name: str, ci_lo: float, ci_hi: float) -> bool:
    """Direction-aware single-metric pass test (VER-07, D-06).

    For ``name`` in :data:`LOWER_IS_BETTER`, passes iff ``ci_hi < 0`` (the WHOLE paired CI lies
    strictly below zero → weatherquant beats v3 on a lower-is-better metric); for ``name`` in
    :data:`HIGHER_IS_BETTER`, passes iff ``ci_lo > 0`` (the whole CI lies strictly above zero). A
    CI that touches or straddles zero is NOT a pass. An unknown metric name fails LOUD (a silent
    default would let an unscored metric pass).
    """
    if name in LOWER_IS_BETTER:
        return ci_hi < 0.0
    if name in HIGHER_IS_BETTER:
        return ci_lo > 0.0
    raise ValueError(
        f"unknown gate-1 metric {name!r}; the pre-registered set is {sorted(GATE1_METRICS)}"
    )


def gate1_passes(cis: Mapping[str, tuple[float, float]]) -> bool:
    """Conjunctive pre-registered Gate-1 verdict over the five pooled metric CIs (VER-07, D-06).

    Asserts the CI dict scores EXACTLY the pre-registered key-set ``{brier, crps, ece, roi, clv}``
    (a missing or extra metric fails loud — the same full-natural-key discipline as
    ``db.queries.latest``, so the bar can't be moved by quietly dropping a metric), then requires
    EVERY metric to pass :func:`metric_passes`. Conjunctive: a single failing metric fails the
    gate. Holm is deliberately NOT applied to these five primaries (the conjunction is conservative
    by construction).
    """
    assert set(cis) == GATE1_METRICS, (
        f"gate-1 verdict must score exactly {sorted(GATE1_METRICS)}, got {sorted(cis)}"
    )
    return all(metric_passes(name, lo, hi) for name, (lo, hi) in cis.items())


def write_preregistration(path: str | Path, spec: Mapping[str, Any]) -> None:
    """Freeze the Gate-1 pre-registration to JSON — refusing to overwrite (irreversible, D-07).

    Writes ``spec`` (metrics list, primary lead, CI level 0.95, n_resamples 10000, RNG seed,
    test-window start/end, the LOWER/HIGHER sign map) to ``path`` as JSON. Refuses to overwrite an
    existing pre-registration: the freeze is irreversible (D-13 in CONTEXT) — re-writing it after a
    run would be the p-hacking vector the guard exists to block. Raises ``FileExistsError`` if the
    artifact already exists.
    """
    p = Path(path)
    if p.exists():
        raise FileExistsError(
            f"pre-registration already frozen at {p}; the Gate-1 endpoint is irreversible "
            f"(D-07) — refusing to overwrite. Delete only to re-register a NEW, un-run gate."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(spec), indent=2, sort_keys=True))


def load_preregistration(path: str | Path) -> dict[str, Any]:
    """Load the frozen Gate-1 pre-registration spec from ``path`` (D-07).

    Returns the spec verbatim as a dict (metrics list, seed, primary lead, test window, sign map,
    …). This is a pure read — it does NOT silently coerce the metric set to the canonical five, so
    a tampered/mismatched pre-registration surfaces to :func:`assert_matches_preregistration`
    rather than being masked.
    """
    return json.loads(Path(path).read_text())


def assert_matches_preregistration(
    spec: Mapping[str, Any], live_params: Mapping[str, Any]
) -> None:
    """Fail LOUD if the live run's parameters differ from the frozen pre-registration (D-07).

    Compares each guarded key (metrics, seed, primary lead, test window, sign map) between the
    frozen ``spec`` and the ``live_params`` actually used. Any divergence raises ``ValueError`` —
    this is the structural anti-p-hacking guard (D-08 in CONTEXT): the Gate-1 bar cannot be moved
    (seed-shopping, lead/window-shifting, metric-swapping) after the results are seen. Metric and
    sign-map sets are compared order-insensitively; scalars are compared by equality.
    """
    mismatches: list[str] = []
    for key in _PREREG_GUARDED_KEYS:
        if key not in spec or key not in live_params:
            continue  # only guard keys present on both sides; absent keys aren't silently equal
        frozen, live = spec[key], live_params[key]
        if key in ("metrics", "sign_map"):
            frozen, live = _normalize_set(frozen), _normalize_set(live)
        if frozen != live:
            mismatches.append(f"{key}: frozen={spec[key]!r} != live={live_params[key]!r}")
    if mismatches:
        raise ValueError(
            "live Gate-1 run does not match the frozen pre-registration (anti-p-hacking, D-07): "
            + "; ".join(mismatches)
        )


def _normalize_set(value: Any) -> Any:
    """Order-insensitive comparison key for the metrics list / sign-map (set of items)."""
    if isinstance(value, Mapping):
        return frozenset(value.items())
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(value)
    return value
