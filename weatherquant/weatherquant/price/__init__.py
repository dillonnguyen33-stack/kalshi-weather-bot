"""Pure-NumPy money path, re-exported as one import site (D-14).

Pure NumPy + stdlib ``math`` only — no scipy/sklearn (AST guard); the normal CDF is reused
from :mod:`weatherquant.calibrate.crps` (D-15). Public surface:

* ``blend``   — ``accuracy_weights``, ``blend_gaussians`` (D-01/D-02/D-03).
* ``buckets`` — ``integers_in_bucket``, ``bucket_prob``, ``bucket_probs`` (D-04/D-05).
* ``ticker``  — ``parse_ticker`` (D-06).
* ``fee``     — ``exact_fee``, ``maker_fee`` (D-07/D-09).
* ``ev``      — ``p_used``, ``bucket_ev`` (D-08).
* ``kelly``   — ``kelly_fraction``, ``sufficiency_ramp``, ``stake_fraction`` (D-10–D-13).
"""

from __future__ import annotations

from weatherquant.price.blend import accuracy_weights, blend_gaussians
from weatherquant.price.buckets import (
    bucket_prob,
    bucket_probs,
    integers_in_bucket,
)
from weatherquant.price.ev import bucket_ev, p_used
from weatherquant.price.fee import exact_fee, maker_fee
from weatherquant.price.kelly import kelly_fraction, stake_fraction, sufficiency_ramp
from weatherquant.price.ticker import parse_ticker

__all__ = [
    # blend (PRC-01, D-01/D-02/D-03)
    "accuracy_weights",
    "blend_gaussians",
    # ev (PRC-03, D-08)
    "bucket_ev",
    # buckets (PRC-02, D-04/D-05/D-06)
    "bucket_prob",
    "bucket_probs",
    # fee (PRC-03, D-07/D-09)
    "exact_fee",
    "integers_in_bucket",
    # kelly (PRC-04/PRC-05, D-10–D-13)
    "kelly_fraction",
    "maker_fee",
    "p_used",
    "parse_ticker",
    "stake_fraction",
    "sufficiency_ramp",
]
