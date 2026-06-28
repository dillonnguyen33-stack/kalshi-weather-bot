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
bucket geometry (never re-implemented).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

# Shared bucket geometry (D-02): BOTH arms price the identical (lo−_HALF, hi+_HALF) spans via the
# same parser/edge helpers — methodology (AS-vs-erf ensemble averaging) is the only difference.
from weatherquant.price.buckets import integers_in_bucket
from weatherquant.price.ticker import parse_ticker

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
    """Legacy v3 normal CDF (Abramowitz-Stegun 7.1.26, 6-dp rounded) — verbatim port (D-02).

    Bit-faithful reproduction of ``kalshi_weather_bot_v3._normal_cdf`` (lines 1319-1326): the
    ``spread == 0`` step, the rational tail approximation with the frozen AS constants, and the
    final ``round(..., 6)``. This is the AS approximation, NOT ``math.erf`` — the head-to-head is
    about methodology, and AS-vs-erf is part of v3's methodology (D-02, golden-tested).
    """
    if spread == 0:
        return 0.0 if x < mean else 1.0
    z = (x - mean) / spread
    t = 1.0 / (1.0 + _AS_T_COEFF * abs(z))
    p = t * (
        _AS_C1 + t * (_AS_C2 + t * (_AS_C3 + t * (_AS_C4 + t * _AS_C5)))
    )
    phi = 1.0 - (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z) * p
    return round(phi if z >= 0 else 1.0 - phi, 6)


def v3_bucket_prob(corrected_mean: float, spread: float, lo: float, hi: float) -> float:
    """Legacy v3 bucket probability (ensemble branch only, D-03) — ``cdf(hi) − cdf(lo)``.

    Mirrors the kind=="B" ENSEMBLE path of ``model_probability`` (lines 1332/1350-1351/1364):
    spread floored at ``0.5``, bucket mass via :func:`v3_normal_cdf`, clamped to ``[0.01, 0.99]``
    and 4-dp rounded. The ``obs_high`` ASOS-override blend, the ``is_next_day`` branch, and the
    threshold (kind!="B") branch are all EXCLUDED (D-03) — there is deliberately no ``obs`` argument.
    """
    spread = max(spread, _V3_SPREAD_FLOOR)
    ensemble_prob = v3_normal_cdf(hi, corrected_mean, spread) - v3_normal_cdf(
        lo, corrected_mean, spread
    )
    return round(max(_V3_PROB_LO, min(_V3_PROB_HI, ensemble_prob)), 4)


def v3_bucket_probs(
    corrected_mean: float,
    spread: float,
    ladder: Sequence[Mapping[str, Any] | str],
) -> dict[str, float]:
    """Legacy v3 bucket probabilities across a full Kalshi ladder (D-02/D-03).

    Prices every bucket on the SHARED geometry the weatherquant arm uses: each ladder entry is
    parsed by :func:`weatherquant.price.ticker.parse_ticker` into ``(lo, hi, open_lo, open_hi)``
    inclusive integer edges, mapped to the continuous ``(lo−_HALF, hi+_HALF)`` span by
    :func:`weatherquant.price.buckets.integers_in_bucket`, then priced by :func:`v3_bucket_prob`.
    BOTH arms therefore share identical bucket geometry — methodology is the only difference (VER-04).

    Each ladder entry is either a ticker string or a mapping carrying ``ticker`` and/or structured
    strikes (``floor_strike``/``cap_strike``/``strike_type``) plus optional ``label``; the returned
    key is the entry's ticker (or its repr when only structured strikes are supplied). Open-tail
    buckets use the ``∓inf`` edge — ``v3_normal_cdf`` returns the legacy step/asymptote there. NO
    same-day obs is read anywhere (D-03 leak guard is structural — no ``obs`` parameter exists).
    """
    out: dict[str, float] = {}
    for entry in ladder:
        ticker, key, parse_kwargs = _ladder_entry_to_parse_args(entry)
        lo_i, hi_i, open_lo, open_hi = parse_ticker(ticker, **parse_kwargs)
        lo_edge, hi_edge = integers_in_bucket(lo_i, hi_i, open_lo=open_lo, open_hi=open_hi)
        out[key] = v3_bucket_prob(corrected_mean, spread, lo_edge, hi_edge)
    return out


def _ladder_entry_to_parse_args(
    entry: Mapping[str, Any] | str,
) -> tuple[str | None, str, dict[str, Any]]:
    """Normalize a ladder entry to ``(ticker, result_key, parse_ticker kwargs)`` (D-02 geometry).

    A bare string is a positional ticker; a mapping forwards ``ticker`` plus the structured strike
    fields (``floor_strike``/``cap_strike``/``strike_type``/``label``) to :func:`parse_ticker`. The
    result key is the ticker when present, else a stable repr of the structured strikes.
    """
    if isinstance(entry, str):
        return entry, entry, {}
    if isinstance(entry, Mapping):
        ticker = entry.get("ticker")
        parse_kwargs = {
            name: entry[name]
            for name in ("floor_strike", "cap_strike", "strike_type", "label")
            if name in entry
        }
        key = (
            str(ticker)
            if ticker is not None
            else "|".join(f"{k}={parse_kwargs[k]}" for k in sorted(parse_kwargs))
        )
        return ticker, key, parse_kwargs
    raise ValueError(
        f"v3_bucket_probs: ladder entry must be a ticker string or a mapping; got {entry!r}."
    )
