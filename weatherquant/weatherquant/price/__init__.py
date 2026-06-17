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

The re-export wiring below is intentionally deferred — these modules carry
``NotImplementedError`` stubs until Waves 1–2 implement them, so importing the names now
would surface stubs rather than the real surface. The actual public re-exports are added in
04-05 once all modules exist.
"""

from __future__ import annotations

__all__: list[str] = []

# TODO (04-05): once Waves 1–2 fill the module bodies, re-export the public surface here so
# callers can ``from weatherquant.price import blend_gaussians, bucket_probs, exact_fee,
# bucket_ev, stake_fraction`` etc. and extend ``__all__`` accordingly:
#   from weatherquant.price.blend import accuracy_weights, blend_gaussians
#   from weatherquant.price.buckets import (
#       bucket_prob, bucket_probs, integers_in_bucket, parse_ticker,
#   )
#   from weatherquant.price.fee import exact_fee, maker_fee
#   from weatherquant.price.ev import bucket_ev, p_used
#   from weatherquant.price.kelly import kelly_fraction, stake_fraction, sufficiency_ramp
