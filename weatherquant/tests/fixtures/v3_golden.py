"""Frozen golden values for the legacy v3 probability numerics (T-06-01 integrity anchor / D-02).

This module is a ONE-TIME verbatim port of the legacy ``kalshi_weather_bot_v3.py`` numerics — the
``_normal_cdf`` Abramowitz-Stegun rational approximation (lines 1319-1326) and the kind=="B"
ENSEMBLE branch of ``model_probability`` (lines 1350-1364: ``cdf(hi) − cdf(lo)``, spread floored at
``0.5``, clamped to ``[0.01, 0.99]`` and 4-dp rounded). The D-03 EXCLUSIONS hold here exactly: NO
intraday ASOS override, NO ``is_next_day`` branch, NO threshold (kind!="B") branch — those v3 paths
are deliberately out of the apples-to-apples Gate-1 comparison.

The ``_legacy_*`` functions below are the verbatim port; the ``V3_*_GOLDEN`` constants are their
outputs frozen over a fixed input grid. ``tests/test_v3_reference.py`` asserts the Wave-2 adapter
``weatherquant.verify.v3_reference`` reproduces every tuple here to the legacy rounding EXACTLY. A
transcription error in the adapter therefore surfaces as a RED test, never a silent verdict drift
(T-06-01). Regenerate by re-running ``_legacy_normal_cdf`` / ``_legacy_bucket_prob`` over the grid.

Constants captured verbatim from the legacy source (do NOT edit):
``0.2316419 / 0.319381530 / -0.356563782 / 1.781477937 / -1.821255978 / 1.330274429``;
6-dp round on the CDF; spread floor ``0.5``; clamp ``[0.01, 0.99]`` + 4-dp round on the bucket.
"""

from __future__ import annotations

import math

# --- Verbatim legacy port (kalshi_weather_bot_v3.py:1319-1364) -------------------------------


def _legacy_normal_cdf(x: float, mean: float, spread: float) -> float:
    """Verbatim port of v3 ``_normal_cdf`` (lines 1319-1326) — do not alter the constants/rounding."""
    if spread == 0:
        return 0.0 if x < mean else 1.0
    z = (x - mean) / spread
    t = 1 / (1 + 0.2316419 * abs(z))
    p = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    phi = 1 - (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * p
    return round(phi if z >= 0 else 1 - phi, 6)


def _legacy_bucket_prob(corrected_mean: float, spread: float, lo: float, hi: float) -> float:
    """Verbatim port of the v3 kind=="B" ENSEMBLE branch (lines 1332/1350-1351/1364), D-03 only.

    Spread floored at ``0.5``; bucket mass ``cdf(hi) − cdf(lo)``; clamped to ``[0.01, 0.99]`` and
    4-dp rounded. NO ASOS override, NO same-day obs, NO threshold branch (D-03 exclusion).
    """
    spread = max(spread, 0.5)
    ensemble_prob = _legacy_normal_cdf(hi, corrected_mean, spread) - _legacy_normal_cdf(
        lo, corrected_mean, spread
    )
    return round(max(0.01, min(0.99, ensemble_prob)), 4)


# --- Frozen golden grids (outputs of the verbatim port above; the adapter must match EXACTLY) ---

# (x, mean, spread, expected_cdf). Spans spread ∈ {0.5, 1.0, 2.5, 5.0} × 3 means × 5 offsets, plus
# the spread==0 step (below/above/at the mean). 6-dp rounded, matching the legacy.
V3_NORMAL_CDF_GOLDEN: list[tuple[float, float, float, float]] = [
    (54.0, 60.0, 0.5, 0.0),
    (58.0, 60.0, 0.5, 3.2e-05),
    (60.0, 60.0, 0.5, 0.5),
    (62.0, 60.0, 0.5, 0.999968),
    (66.0, 60.0, 0.5, 1.0),
    (66.5, 72.5, 0.5, 0.0),
    (70.5, 72.5, 0.5, 3.2e-05),
    (72.5, 72.5, 0.5, 0.5),
    (74.5, 72.5, 0.5, 0.999968),
    (78.5, 72.5, 0.5, 1.0),
    (79.0, 85.0, 0.5, 0.0),
    (83.0, 85.0, 0.5, 3.2e-05),
    (85.0, 85.0, 0.5, 0.5),
    (87.0, 85.0, 0.5, 0.999968),
    (91.0, 85.0, 0.5, 1.0),
    (54.0, 60.0, 1.0, 0.0),
    (58.0, 60.0, 1.0, 0.02275),
    (60.0, 60.0, 1.0, 0.5),
    (62.0, 60.0, 1.0, 0.97725),
    (66.0, 60.0, 1.0, 1.0),
    (66.5, 72.5, 1.0, 0.0),
    (70.5, 72.5, 1.0, 0.02275),
    (72.5, 72.5, 1.0, 0.5),
    (74.5, 72.5, 1.0, 0.97725),
    (78.5, 72.5, 1.0, 1.0),
    (79.0, 85.0, 1.0, 0.0),
    (83.0, 85.0, 1.0, 0.02275),
    (85.0, 85.0, 1.0, 0.5),
    (87.0, 85.0, 1.0, 0.97725),
    (91.0, 85.0, 1.0, 1.0),
    (54.0, 60.0, 2.5, 0.008198),
    (58.0, 60.0, 2.5, 0.211855),
    (60.0, 60.0, 2.5, 0.5),
    (62.0, 60.0, 2.5, 0.788145),
    (66.0, 60.0, 2.5, 0.991802),
    (66.5, 72.5, 2.5, 0.008198),
    (70.5, 72.5, 2.5, 0.211855),
    (72.5, 72.5, 2.5, 0.5),
    (74.5, 72.5, 2.5, 0.788145),
    (78.5, 72.5, 2.5, 0.991802),
    (79.0, 85.0, 2.5, 0.008198),
    (83.0, 85.0, 2.5, 0.211855),
    (85.0, 85.0, 2.5, 0.5),
    (87.0, 85.0, 2.5, 0.788145),
    (91.0, 85.0, 2.5, 0.991802),
    (54.0, 60.0, 5.0, 0.11507),
    (58.0, 60.0, 5.0, 0.344578),
    (60.0, 60.0, 5.0, 0.5),
    (62.0, 60.0, 5.0, 0.655422),
    (66.0, 60.0, 5.0, 0.88493),
    (66.5, 72.5, 5.0, 0.11507),
    (70.5, 72.5, 5.0, 0.344578),
    (72.5, 72.5, 5.0, 0.5),
    (74.5, 72.5, 5.0, 0.655422),
    (78.5, 72.5, 5.0, 0.88493),
    (79.0, 85.0, 5.0, 0.11507),
    (83.0, 85.0, 5.0, 0.344578),
    (85.0, 85.0, 5.0, 0.5),
    (87.0, 85.0, 5.0, 0.655422),
    (91.0, 85.0, 5.0, 0.88493),
    (71.0, 72.0, 0.0, 0.0),
    (73.0, 72.0, 0.0, 1.0),
    (72.0, 72.0, 0.0, 1.0),
]

# (corrected_mean, spread, lo, hi, expected_bucket_prob). The first block has spread 0.3 < 0.5 so
# it exercises the legacy spread FLOOR (0.3 → 0.5). 4-dp rounded, clamped to [0.01, 0.99].
V3_BUCKET_PROB_GOLDEN: list[tuple[float, float, float, float, float]] = [
    (70.0, 0.3, 69.0, 71.0, 0.9545),
    (70.0, 0.3, 71.0, 73.0, 0.0228),
    (70.0, 0.3, 67.0, 69.0, 0.0227),
    (70.0, 0.3, 75.0, 78.0, 0.01),
    (78.0, 0.3, 77.0, 79.0, 0.9545),
    (78.0, 0.3, 79.0, 81.0, 0.0228),
    (78.0, 0.3, 75.0, 77.0, 0.0227),
    (78.0, 0.3, 83.0, 86.0, 0.01),
    (70.0, 0.5, 69.0, 71.0, 0.9545),
    (70.0, 0.5, 71.0, 73.0, 0.0228),
    (70.0, 0.5, 67.0, 69.0, 0.0227),
    (70.0, 0.5, 75.0, 78.0, 0.01),
    (78.0, 0.5, 77.0, 79.0, 0.9545),
    (78.0, 0.5, 79.0, 81.0, 0.0228),
    (78.0, 0.5, 75.0, 77.0, 0.0227),
    (78.0, 0.5, 83.0, 86.0, 0.01),
    (70.0, 1.5, 69.0, 71.0, 0.495),
    (70.0, 1.5, 71.0, 73.0, 0.2297),
    (70.0, 1.5, 67.0, 69.0, 0.2297),
    (70.0, 1.5, 75.0, 78.0, 0.01),
    (78.0, 1.5, 77.0, 79.0, 0.495),
    (78.0, 1.5, 79.0, 81.0, 0.2297),
    (78.0, 1.5, 75.0, 77.0, 0.2297),
    (78.0, 1.5, 83.0, 86.0, 0.01),
    (70.0, 3.0, 69.0, 71.0, 0.2611),
    (70.0, 3.0, 71.0, 73.0, 0.2108),
    (70.0, 3.0, 67.0, 69.0, 0.2108),
    (70.0, 3.0, 75.0, 78.0, 0.044),
    (78.0, 3.0, 77.0, 79.0, 0.2611),
    (78.0, 3.0, 79.0, 81.0, 0.2108),
    (78.0, 3.0, 75.0, 77.0, 0.2108),
    (78.0, 3.0, 83.0, 86.0, 0.044),
]


def _regenerate() -> tuple[list, list]:
    """Recompute both golden grids from the verbatim port (used to refresh the frozen constants)."""
    cdf = [(x, m, s, _legacy_normal_cdf(x, m, s)) for (x, m, s, _e) in V3_NORMAL_CDF_GOLDEN]
    bucket = [
        (cm, s, lo, hi, _legacy_bucket_prob(cm, s, lo, hi))
        for (cm, s, lo, hi, _e) in V3_BUCKET_PROB_GOLDEN
    ]
    return cdf, bucket
