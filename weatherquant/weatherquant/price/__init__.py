"""Pure-NumPy money path: blend → buckets → fee → EV → fractional Kelly (Phase 4, D-14).

``weatherquant.price`` turns Phase-3 calibrated per-model Gaussians into a single blended
predictive distribution, Kalshi bucket probabilities, fee-corrected expected value, and a
capped fractional-Kelly stake. It is a pure compute library — market prices are inputs and
no live market I/O happens here; trade intents are persisted at the Phase-5 boundary (D-16).

The whole package is **pure NumPy + stdlib ``math``** — no scipy/sklearn anywhere (D-14);
the cloned AST guard ``tests/test_no_forbidden_price_deps.py`` fences those out, and the
normal CDF is reused verbatim from :mod:`weatherquant.calibrate.crps` (one source of truth,
D-15) rather than re-implemented here.

Public surface (wired in 04-05 once every module is filled in Waves 1–2):

* ``blend``   — ``accuracy_weights``, ``blend_gaussians`` (Vincentization closed form,
  D-01/D-02/D-03).
* ``buckets`` — ``parse_ticker``, ``integers_in_bucket``, ``bucket_prob``, ``bucket_probs``
  (CDF differencing over the integer-°F ladder, D-04/D-05/D-06).
* ``fee``     — ``exact_fee``, ``maker_fee`` (exact integer-cent Kalshi fee, D-07/D-09).
* ``ev``      — ``p_used``, ``bucket_ev`` (market-shrunk fee-corrected EV, D-08).
* ``kelly``   — ``kelly_fraction``, ``sufficiency_ramp``, ``stake_fraction`` (fee-aware
  fractional Kelly with σ/sufficiency/AFD shrink and a hard position cap, D-10–D-13).

Now that every module is filled in (04-02 blend, 04-03 buckets, 04-04 fee/ev, 04-05 kelly),
the full public surface is re-exported here so callers can do
``from weatherquant.price import blend_gaussians, bucket_probs, exact_fee, bucket_ev,
stake_fraction`` etc. — a single import site for the whole money path. ``__all__`` lists the
full surface; the CLI ``price`` smoke command (the I/O edge) consumes it.
"""

from __future__ import annotations

from weatherquant.price.blend import accuracy_weights, blend_gaussians
from weatherquant.price.buckets import (
    bucket_prob,
    bucket_probs,
    integers_in_bucket,
    parse_ticker,
)
from weatherquant.price.ev import bucket_ev, p_used
from weatherquant.price.fee import exact_fee, maker_fee
from weatherquant.price.kelly import kelly_fraction, stake_fraction, sufficiency_ramp

__all__ = [
    # blend (PRC-01, D-01/D-02/D-03)
    "accuracy_weights",
    "blend_gaussians",
    # buckets (PRC-02, D-04/D-05/D-06)
    "bucket_prob",
    "bucket_probs",
    "integers_in_bucket",
    "parse_ticker",
    # fee (PRC-03, D-07/D-09)
    "exact_fee",
    "maker_fee",
    # ev (PRC-03, D-08)
    "bucket_ev",
    "p_used",
    # kelly (PRC-04/PRC-05, D-10–D-13)
    "kelly_fraction",
    "sufficiency_ramp",
    "stake_fraction",
]
