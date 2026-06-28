"""Gate-1 metric core (VER-01): Brier, ECE, CRPS, ROI, CLV.

D-07 (verify subtree-local): the proof metrics are hand-rolled NumPy + stdlib — no scipy/sklearn
(fenced by ``tests/test_no_forbidden_verify_deps.py``). The Gaussian CRPS and the erf-based normal
CDF are REUSED from :mod:`weatherquant.calibrate.crps` (the one source of truth, D-04) — never
re-derived here — so ``crps_blend`` agrees with the calibration core by construction.

D-06 (verify subtree-local): one scalar per Gate-1 metric. The ECE *scalar* uses EQUAL-COUNT
(equal-mass) bins so sparse high-confidence regions are weighted by their actual population, not
their (empty) width — kept distinct from the EQUAL-WIDTH reliability-diagram grid in ``report.py``
(RESEARCH §Pitfall 4).

D-01 (verify subtree-local): pure-NumPy metric core — fail-loud at every cents/probability/dollars
boundary (RESEARCH §Pitfall 3): probabilities ∈ [0, 1], σ > 0, prices ∈ [0, 100] cents. Never
silently coerce — a unit mismatch (ROI off by ~100×, a probability > 1, a price > 100) must raise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

# D-04 single source of truth: reuse the one closed-form Gaussian CRPS + erf-based CDF rather than
# re-deriving them here (re-deriving would change the weatherquant arm and void the pre-registered
# comparison — RESEARCH §Don't Hand-Roll).
from weatherquant.calibrate.crps import crps_norm
from weatherquant.market.clv import clv_cents

__all__ = [
    "brier",
    "ece_equal_count",
    "crps_blend",
    "roi_from_fills",
    "mean_clv",
]

# Default bin count for the equal-count ECE (overridable per call).
_DEFAULT_N_BINS = 10


def _require_prob_array(name: str, values: NDArray[np.float64]) -> None:
    """Fail loud if any probability falls outside ``[0, 1]`` or is non-finite (D-01, ASVS V5)."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite; got non-finite entries.")
    if values.size and (values.min() < 0.0 or values.max() > 1.0):
        raise ValueError(f"{name} must be in [0, 1]; got [{values.min()}, {values.max()}].")


def _require_binary(name: str, values: NDArray[np.float64]) -> None:
    """Fail loud if any outcome is not in ``{0, 1}`` (D-01) — outcomes are realized YES/NO."""
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite; got non-finite entries.")
    if values.size and not np.all((values == 0.0) | (values == 1.0)):
        raise ValueError(f"{name} must be binary {{0, 1}}; got other values.")


def brier(f: NDArray[np.float64], o: NDArray[np.float64]) -> float:
    """Mean Brier score ``mean((f - o)^2)`` over forecast probabilities ``f`` and outcomes ``o``.

    ``f`` are predicted YES probabilities in ``[0, 1]``; ``o`` are realized ``{0, 1}`` outcomes.
    Lower is better — the bin-free, exact Gate-1 scalar (D-06). Guards ``f ∈ [0, 1]`` and binary
    ``o`` (D-01).
    """
    f = np.asarray(f, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)
    _require_prob_array("brier f", f)
    _require_binary("brier o", o)
    return float(np.mean((f - o) ** 2))


def ece_equal_count(
    f: NDArray[np.float64], o: NDArray[np.float64], n_bins: int = _DEFAULT_N_BINS
) -> float:
    """Expected Calibration Error over ``n_bins`` EQUAL-COUNT (quantile) bins (VER-01, D-06).

    ``ECE = Σ_k (n_k/N)·|mean(o_k) - mean(f_k)|`` over equal-mass bins (``np.array_split`` of the
    forecast argsort) — NOT equal-width — so sparse high-confidence bins are weighted by their
    actual population, never their empty width (RESEARCH §Pitfall 4). ~0 for a perfectly
    calibrated forecast, clearly > 0 for a biased one. Guards ``f ∈ [0, 1]``, binary ``o`` (D-01).
    """
    f = np.asarray(f, dtype=np.float64)
    o = np.asarray(o, dtype=np.float64)
    _require_prob_array("ece f", f)
    _require_binary("ece o", o)

    n = len(f)
    if n == 0:
        return 0.0
    order = np.argsort(f)
    f_sorted = f[order]
    o_sorted = o[order]
    ece = 0.0
    for b in np.array_split(np.arange(n), n_bins):
        if b.size == 0:
            continue
        conf = float(f_sorted[b].mean())
        acc = float(o_sorted[b].mean())
        ece += (b.size / n) * abs(acc - conf)
    return float(ece)


def crps_blend(
    mu: NDArray[np.float64], sigma: NDArray[np.float64], y: NDArray[np.float64]
) -> float:
    """Mean Gaussian CRPS of the blended predictive against the verifying obs (VER-01).

    The Gate-1 blend is a SINGLE Gaussian (RESEARCH §Summary 1), so the D-06 "sample/quantile-CRPS
    otherwise" branch is dead code: this ASSERTS the blend is Gaussian (``sigma > 0``) and delegates
    elementwise to :func:`weatherquant.calibrate.crps.crps_norm` (the one closed-form source, D-04),
    returning the mean — never a re-derived or sample-CRPS fallback. Re-deriving would change the
    weatherquant arm and void the pre-registered comparison.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    # The blend is Gaussian by construction — assert it (NO sample-CRPS fallback is built, D-06).
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0.0):
        raise ValueError(
            "crps_blend requires a Gaussian blend (sigma finite and > 0 elementwise) — "
            "the single-Gaussian blend has no sample-CRPS fallback (D-06)."
        )
    return float(crps_norm(mu, sigma, y).mean())


def roi_from_fills(
    fills: Sequence[Any],
    settled_yes: Sequence[bool],
    sides: Sequence[str],
) -> float:
    """Realized return-on-investment over paper fills given per-market YES settlement (VER-01).

    SIDE MIRROR (mirrors :func:`weatherquant.market.clv.clv_cents`' D-09 sign orientation): the
    payoff is oriented by the fill's ``side``. A YES-equivalent BUY (``side`` in ``"yes"``/``"buy"``)
    wins when the bucket settles YES; a NO-equivalent position (``side`` in ``"no"``/``"sell"``) is
    the MIRROR — it wins when the bucket settles NO. ``side`` is normalized to buy/sell exactly as
    ``cli.verify._settle_window_fills`` does (``"yes"``/``"buy"`` → buy, ``"no"``/``"sell"`` → sell);
    a side-blind ROI would mis-score a NO win as a loss (the 06-09 defect).

    D-01 unit discipline (RESEARCH §Pitfall 3): everything in CENTS, dollars only at the return
    edge. Per fill ``i`` with ``count`` contracts and the FLOAT ``detail['avg_price_cents']``
    (never the ±0.5c-rounded ``fills.price``):

        entry_i  = count_i · avg_price_cents_i                       # capital deployed, cents
        wins_i   = settled_yes_i        if side is YES-equivalent     # else: NOT settled_yes_i
        payoff_i = 100 · count_i  if wins_i else 0                    # 100c per winning contract
        net_i    = payoff_i − entry_i − fee_i                        # fee in cents

    ``ROI = Σ net / Σ entry`` (a pure ratio — unit-free). Guards ``avg_price_cents ∈ [0, 100]``
    (a price > 100 or < 0 is a unit bug, D-01).
    """
    if not (len(fills) == len(settled_yes) == len(sides)):
        raise ValueError(
            f"roi_from_fills: fills/settled_yes/sides length mismatch "
            f"({len(fills)}, {len(settled_yes)}, {len(sides)})."
        )
    total_entry = 0.0
    total_net = 0.0
    for fill, is_yes, side in zip(fills, settled_yes, sides, strict=True):
        count = float(_fill_field(fill, "count"))
        avg_price_cents = float(_fill_avg_price_cents(fill))
        fee = float(_fill_field(fill, "fee"))
        if not (0.0 <= avg_price_cents <= 100.0):
            raise ValueError(
                f"roi_from_fills: avg_price_cents must be in [0, 100] cents; "
                f"got {avg_price_cents} (unit bug — dollars vs cents?)."
            )
        # Normalize to buy/sell exactly as cli.verify._settle_window_fills does, then mirror the
        # clv_cents orientation: a YES-equivalent BUY wins on a YES settlement, a NO-equivalent
        # position wins on a NO settlement.
        is_buy = side in ("yes", "buy")
        wins = is_yes if is_buy else (not is_yes)
        entry = count * avg_price_cents
        payoff = 100.0 * count if wins else 0.0
        total_entry += entry
        total_net += payoff - entry - fee
    if total_entry <= 0.0:
        raise ValueError(
            "roi_from_fills: total capital deployed is non-positive — no ROI is defined "
            "(empty or zero-cost fill set)."
        )
    return total_net / total_entry


def mean_clv(
    fills: Sequence[Any],
    closing_snapshots: Sequence[Sequence[Mapping[str, Any]]],
    sides: Sequence[str],
) -> float:
    """Mean Closing-Line Value (CENTS) over the paper fills vs the closing volume-weighted mid.

    Delegates per fill to :func:`weatherquant.market.clv.clv_cents` (D-12 derived-CLV convention,
    reused verbatim — it reads the FLOAT ``avg_price_cents`` and never the rounded ``price``), then
    returns the arithmetic mean in cents. ``closing_snapshots[i]`` is the closing-window snapshot
    sequence for fill ``i`` (typically ``market.clv.closing_window_snapshots`` output) and
    ``sides[i]`` is ``"buy"``/``"sell"``. Positive mean CLV is the edge-vs-close signal.
    """
    if not (len(fills) == len(closing_snapshots) == len(sides)):
        raise ValueError(
            f"mean_clv: fills/closing_snapshots/sides length mismatch "
            f"({len(fills)}, {len(closing_snapshots)}, {len(sides)})."
        )
    if len(fills) == 0:
        raise ValueError("mean_clv: no fills — mean CLV is undefined on an empty set.")
    clvs = [
        clv_cents(fill, snaps, side)
        for fill, snaps, side in zip(fills, closing_snapshots, sides, strict=True)
    ]
    return float(np.mean(clvs))


def _fill_field(fill: Any, name: str) -> Any:
    """Read a fill attribute or mapping key, fail loud if absent (D-01 — no silent default)."""
    if isinstance(fill, Mapping):
        if name not in fill:
            raise ValueError(f"fill is missing required field {name!r}.")
        return fill[name]
    if hasattr(fill, name):
        return getattr(fill, name)
    raise ValueError(f"fill is missing required field {name!r}.")


def _fill_avg_price_cents(fill: Any) -> float:
    """Read the FLOAT ``avg_price_cents`` (never the rounded ``price``) from a fill (D-01/§Pitfall 3).

    Supports an attribute ``avg_price_cents`` (the Phase-5 fill object) or a ``detail`` mapping
    carrying ``avg_price_cents`` (the ledger row shape). Fails loud if neither is present.
    """
    if hasattr(fill, "avg_price_cents"):
        return float(fill.avg_price_cents)
    detail = fill.get("detail") if isinstance(fill, Mapping) else getattr(fill, "detail", None)
    if isinstance(detail, Mapping) and "avg_price_cents" in detail:
        return float(detail["avg_price_cents"])
    raise ValueError(
        "fill carries no float avg_price_cents (attribute or detail['avg_price_cents']) — "
        "refusing to fall back to the rounded price (±0.5c bias, §Pitfall 3)."
    )
