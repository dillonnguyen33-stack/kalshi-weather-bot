"""Legacy v3 probability adapter (VER-04 / D-02 / D-03): reproduce v3's bucket probabilities exactly.

D-02 (verify subtree-local): this adapter reproduces the legacy ``kalshi_weather_bot_v3.py``
``_normal_cdf`` (Abramowitz-Stegun rational approximation, 6-dp rounded) and the kind=="B"
ENSEMBLE branch of ``model_probability`` (bucket = ``cdf(hi) − cdf(lo)``, spread floored at 0.5,
clamped to ``[0.01, 0.99]`` and 4-dp rounded) so the v3 baseline in the paired backtest is the
TRUE legacy number, not a re-derivation. It is golden-tested against ``tests/fixtures/v3_golden.py``
(a one-time verbatim port) — a transcription error surfaces as a RED test, never a silent verdict.

D-03 EXCLUSION (the leak guard): this adapter is the pure ENSEMBLE math ONLY. It does NOT read
same-day ASOS observations, has NO intraday ASOS override, and NO ``is_next_day`` / threshold
branch — those v3 paths are excluded from the apples-to-apples Gate-1 comparison. There is
deliberately no ``obs`` argument in any signature here.

``math`` only for the numerics; ``parse_ticker`` / ``integers_in_bucket`` are reused for shared
bucket geometry (never re-implemented). Bodies land Wave 2; ``tests/test_v3_reference.py`` (RED).
"""

from __future__ import annotations

# ``math`` + the shared bucket geometry are imported now (the Wave-2 numeric port reuses them
# rather than re-implementing erf or the bucket edges); referenced once the stubs land.
import math  # noqa: F401  (Wave-2 seam)

from weatherquant.price.buckets import integers_in_bucket  # noqa: F401  (Wave-2 seam)
from weatherquant.price.ticker import parse_ticker  # noqa: F401  (Wave-2 seam)

__all__ = ["v3_normal_cdf", "v3_bucket_prob", "v3_bucket_probs"]

# Legacy Abramowitz-Stegun 7.1.26 rational-approximation constants (verbatim from v3
# _normal_cdf, lines 1319-1326). Frozen here so the adapter matches the legacy rounding exactly.
_AS_T_COEFF = 0.2316419
_AS_C1 = 0.319381530
_AS_C2 = -0.356563782
_AS_C3 = 1.781477937
_AS_C4 = -1.821255978
_AS_C5 = 1.330274429
# Legacy spread floor and probability clamp (model_probability lines 1332/1364).
_V3_SPREAD_FLOOR = 0.5
_V3_PROB_LO = 0.01
_V3_PROB_HI = 0.99


def v3_normal_cdf(x: float, mean: float, spread: float) -> float:
    """Legacy v3 normal CDF (Abramowitz-Stegun, 6-dp rounded) — verbatim numeric port (D-02).

    Reproduces ``kalshi_weather_bot_v3._normal_cdf`` exactly, including the ``spread == 0`` step,
    the rational tail approximation, and the final ``round(..., 6)``. Body lands Wave 2.
    """
    raise NotImplementedError("verify.v3_reference.v3_normal_cdf lands in Wave 2 (VER-04).")


def v3_bucket_prob(corrected_mean: float, spread: float, lo: float, hi: float) -> float:
    """Legacy v3 bucket probability (ensemble branch only, D-03) — ``cdf(hi) − cdf(lo)``.

    Mirrors the kind=="B" ENSEMBLE path of ``model_probability``: spread floored at ``0.5``, the
    bucket mass via :func:`v3_normal_cdf`, clamped to ``[0.01, 0.99]`` and 4-dp rounded. NO ASOS
    override, NO same-day obs, NO threshold branch (D-03). Body lands Wave 2.
    """
    raise NotImplementedError("verify.v3_reference.v3_bucket_prob lands in Wave 2 (VER-04).")


def v3_bucket_probs(corrected_mean: float, spread: float, ladder) -> dict:
    """Legacy v3 bucket probabilities across a ladder (D-02/D-03).

    ``ladder`` carries the integer-°F bucket edges; reuses ``parse_ticker`` / ``integers_in_bucket``
    for the shared geometry and :func:`v3_bucket_prob` per bucket. Returns a per-bucket mapping.
    Body lands Wave 2.
    """
    raise NotImplementedError("verify.v3_reference.v3_bucket_probs lands in Wave 2 (VER-04).")
