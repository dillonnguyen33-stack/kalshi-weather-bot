"""Fee-aware fractional Kelly with σ/sufficiency/AFD shrink and a hard cap (PRC-04/05).

Stake = ``clip( KELLY_LAMBDA · f_kelly · shrink(σ_blend, sufficiency, afd), 0, cap )``
(D-10–D-13). ``f_kelly`` is the fee-aware Kelly fraction on the positive-EV side only (else
0), recomputed with the EXACT taker fee. ``shrink ∈ (0, 1]`` multiplies three confidence
factors: a smooth σ-shrink ``1/(1 + σ_blend/SIGMA0_F)`` (a vague forecast bets less, D-11),
a sufficiency ramp on Phase-3 ``n_train``/``pool_level`` (thin/pooled data bets less, D-11),
and the AFD haircut ``AFD_HAIRCUT`` when the forecaster-disagreement flag is set — a SOFT
multiplicative cut ``<1`` that reduces but NEVER zeroes the stake (D-12, PRC-05). The final
``clip`` to ``cap`` is a hard invariant: no sized position ever exceeds the configured cap
(D-13), asserted for arbitrary inputs by ``tests/test_kelly.py -k cap``.

All knobs (``KELLY_LAMBDA``, ``SIGMA0_F``, ``AFD_HAIRCUT``, ``N_REF``) are named module
constants with docstrings citing RESEARCH Operational Defaults — never a magic number inline
— so Phase 6 can audit and (within its pre-registered process) revisit them. The bankroll and
the position-cap fraction are typed config on ``weatherquant.db.engine.Settings`` (D-13).

Pure NumPy + stdlib ``math`` only — no scipy/sklearn (the AST guard enforces it).
"""

from __future__ import annotations

import math

from weatherquant.db.engine import DEFAULT_POSITION_FRACTION  # single source for the cap default (WR-A1)
from weatherquant.price.ev import _require_prob  # one prob/price validator for the whole money path
from weatherquant.price.fee import exact_fee  # fee-aware Kelly reuses the exact integer-cent fee

__all__ = ["KELLY_LAMBDA", "SIGMA0_F", "AFD_HAIRCUT", "N_REF",
           "kelly_fraction", "sufficiency_ramp", "stake_fraction"]

# Base fractional-Kelly multiplier (D-10, RESEARCH Operational Defaults): quarter-Kelly is
# the standard conservative choice — full Kelly is high-variance and amplifies prob error.
KELLY_LAMBDA = 0.25

# σ-shrink scale in °F (D-11, RESEARCH Operational Defaults): shrink = 1/(1 + σ_blend/σ₀),
# σ₀≈5 set near a typical daily-high σ so a very wide spread shrinks the bet to ~0.5×.
SIGMA0_F = 5.0

# AFD forecaster-disagreement haircut (D-12, RESEARCH Operational Defaults): a SOFT
# multiplicative cut within CONTEXT's 0.5–0.7 band — reduces stake, never zeroes it (PRC-05).
AFD_HAIRCUT = 0.6

# Sufficiency-ramp reference sample count (D-11): ties to Phase-3's N_MIN=30 so a fit with
# n_train ≥ N_REF is "sufficient" (ramp = 1) and thinner fits bet proportionally less.
N_REF = 30


def kelly_fraction(p: float, price: float, fee: float | None = None) -> float:
    """Fee-aware Kelly fraction on one side; 0 on non-positive edge (D-10).

    ``win = (1 − price) − fee``; if ``win ≤ 0`` returns 0. Otherwise
    ``max(0, (p·win − (1 − p)·price) / win)``. ``tests/test_kelly.py -k zero`` asserts a
    non-positive-EV side returns 0.

    ``fee`` is the EXACT taker fee for the marginal contract. When omitted it is resolved
    here via :func:`weatherquant.price.fee.exact_fee` (the per-contract marginal
    ``exact_fee(1, price)``, Open Question 3) so the sizing path is fee-aware through the ONE
    fee source of truth and never re-implements the fee (D-09); a caller that already has the
    fee (e.g. the EV path) passes it in to avoid recomputing.
    """
    # Fail-loud guards (WR-03): match the rest of price/ (bucket_prob/exact_fee/p_used). Without
    # these, p>1 makes (1−p) negative and silently inflates the edge — a latent money-path footgun.
    _require_prob("p", p)
    _require_prob("price", price)
    if fee is None:
        fee = exact_fee(1, price)  # per-contract marginal taker fee via the one fee seam (D-09)
    if not math.isfinite(fee):
        raise ValueError(f"fee must be finite, got {fee!r}.")
    win = (1.0 - price) - fee
    if win <= 0.0:  # no net upside after fee → non-positive-EV side sizes to 0 (D-10)
        return 0.0
    edge = p * win - (1.0 - p) * price
    return max(0.0, edge / win)


def sufficiency_ramp(n_train: int, pool_level: str) -> float:
    """Confidence ramp from Phase-3 sample count + pooling provenance (D-11 — Wave 2).

    ``min(1, n_train / N_REF)`` with an extra ×0.7 when ``pool_level`` starts ``parent:``
    (a parent-pooled fit is less trustworthy than an own-stratum fit). Thinner/pooled data
    ⇒ smaller stake (D-11); ``tests/test_kelly.py -k shrink`` guards monotonicity.
    """
    ramp = min(1.0, max(0.0, n_train / N_REF))
    if pool_level.startswith("parent:"):
        # A parent-pooled fit is less trustworthy than an own-stratum fit — an extra haircut
        # (D-11, RESEARCH Operational Defaults) so a pure-parent stratum bets more cautiously.
        ramp *= 0.7
    return ramp


def stake_fraction(
    p: float,
    price: float,
    fee: float,
    sigma_blend: float,
    n_train: int,
    pool_level: str,
    afd_flag: bool,
    *,
    lam: float = KELLY_LAMBDA,
    cap: float = DEFAULT_POSITION_FRACTION,
    sigma0: float = SIGMA0_F,
    afd_haircut: float = AFD_HAIRCUT,
) -> float:
    """Capped fractional-Kelly stake fraction with σ/sufficiency/AFD shrink (D-10–D-13 — Wave 2).

    ``clip( lam · kelly_fraction(p, price, fee) · shrink, 0, cap )`` where
    ``shrink = 1/(1 + sigma_blend/sigma0) · sufficiency_ramp(n_train, pool_level) ·
    (afd_haircut if afd_flag else 1.0)``. The AFD haircut is soft (never zeroes — PRC-05);
    the final ``cap`` is a hard invariant (no position exceeds it — D-13), asserted by
    ``tests/test_kelly.py -k cap`` / ``-k afd``.

    The ``cap`` is the position-cap fraction the caller threads through from
    ``Settings.max_position_fraction`` (bounds-validated to ``[0.02, 0.05]`` in 04-01); it
    defaults to the shared ``db.engine.DEFAULT_POSITION_FRACTION`` — the SAME constant that
    field default references — only when no caller supplies it (WR-A1: one source of truth,
    so the default path can't drift from the configured cap). The final
    ``min(max(..., 0.0), cap)`` is the LAST operation and the tested hard invariant: no sized
    position ever exceeds ``cap`` for any input (D-13, threat T-04-13).
    """
    # Fail-loud on a degenerate σ (WR-03): sigma_blend ≤ 0 (e.g. −sigma0) divides by zero or
    # turns s_sigma into a large pre-cap amplifier; a real blend σ from blend_gaussians is > 0.
    if not math.isfinite(sigma_blend):
        raise ValueError(f"sigma_blend must be finite, got {sigma_blend!r}.")
    if sigma_blend <= 0.0:
        raise ValueError(f"sigma_blend must be > 0, got {sigma_blend!r}.")
    f = kelly_fraction(p, price, fee)
    s_sigma = 1.0 / (1.0 + sigma_blend / sigma0)  # vaguer blend (wider σ) ⇒ smaller bet (D-11)
    s_suff = sufficiency_ramp(n_train, pool_level)  # thin/pooled data ⇒ smaller bet (D-11)
    # AFD is a SOFT multiplicative haircut < 1 — it strictly reduces the stake but NEVER
    # zeroes it (D-12, PRC-05); a hard gate here would violate the requirement (threat T-04-14).
    s_afd = afd_haircut if afd_flag else 1.0
    shrink = s_sigma * s_suff * s_afd
    # Hard cap is the final clip (D-13): the last operation, asserted for arbitrary inputs.
    return min(max(lam * f * shrink, 0.0), cap)
