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


def kelly_fraction(p: float, price: float, fee: float) -> float:
    """Fee-aware Kelly fraction on one side; 0 on non-positive edge (D-10 — Wave 2).

    ``win = (1 − price) − fee``; if ``win ≤ 0`` returns 0. Otherwise
    ``max(0, (p·win − (1 − p)·price) / win)``. Recomputed with the EXACT taker fee.
    ``tests/test_kelly.py -k zero`` asserts a non-positive-EV side returns 0.
    """
    raise NotImplementedError("kelly_fraction is implemented in Wave 2 (04-06).")


def sufficiency_ramp(n_train: int, pool_level: str) -> float:
    """Confidence ramp from Phase-3 sample count + pooling provenance (D-11 — Wave 2).

    ``min(1, n_train / N_REF)`` with an extra ×0.7 when ``pool_level`` starts ``parent:``
    (a parent-pooled fit is less trustworthy than an own-stratum fit). Thinner/pooled data
    ⇒ smaller stake (D-11); ``tests/test_kelly.py -k shrink`` guards monotonicity.
    """
    raise NotImplementedError("sufficiency_ramp is implemented in Wave 2 (04-06).")


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
    cap: float = 0.025,
    sigma0: float = SIGMA0_F,
    afd_haircut: float = AFD_HAIRCUT,
) -> float:
    """Capped fractional-Kelly stake fraction with σ/sufficiency/AFD shrink (D-10–D-13 — Wave 2).

    ``clip( lam · kelly_fraction(p, price, fee) · shrink, 0, cap )`` where
    ``shrink = 1/(1 + sigma_blend/sigma0) · sufficiency_ramp(n_train, pool_level) ·
    (afd_haircut if afd_flag else 1.0)``. The AFD haircut is soft (never zeroes — PRC-05);
    the final ``cap`` is a hard invariant (no position exceeds it — D-13), asserted by
    ``tests/test_kelly.py -k cap`` / ``-k afd``.
    """
    raise NotImplementedError("stake_fraction is implemented in Wave 2 (04-06).")
