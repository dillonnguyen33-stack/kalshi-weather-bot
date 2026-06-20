"""Blended CDF → Kalshi bucket probabilities + ticker parser (D-04/D-05/D-06).

CDF differencing ``P(bucket) = Φ_blend(upper) − Φ_blend(lower)`` over the integer-°F ladder,
reusing the erf-based ``normal_cdf`` from :mod:`weatherquant.calibrate.crps` (D-04, Pitfall 6);
open buckets use the one-sided tail. Each integer degree ``k`` owns ``[k − _HALF, k + _HALF)``
(D-05, Pitfall 1). The exact label coverage is LOW-confidence; the live KXHIGH cross-check is
DEFERRED to Phase 5 (D-05; see docs/DECISIONS.md), and the principled mapping is retained
as-is. ``parse_ticker`` is a pure, fail-loud string→edges parser (D-06).

Pure NumPy + stdlib ``math`` only — no scipy/sklearn (AST guard).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from weatherquant.calibrate.crps import normal_cdf

__all__ = ["bucket_prob", "bucket_probs", "integers_in_bucket", "parse_ticker"]

# The single half-degree bucket-edge offset (D-05 / Pitfall 1): integer degree ``k`` owns
# ``[k − _HALF, k + _HALF)``. LOW-confidence; the live KXHIGH cross-check is DEFERRED to
# Phase 5 (see docs/DECISIONS.md) — do not treat as confirmed-against-live.
_HALF = 0.5


def integers_in_bucket(
    lo: int | None,
    hi: int | None,
    open_lo: bool = False,
    open_hi: bool = False,
) -> tuple[float, float]:
    """Continuous ``[lo − _HALF, hi + _HALF)`` span for a labeled bucket (D-05, Pitfall 1).

    Maps the inclusive integer degrees ``lo``..``hi`` to the single continuous interval for CDF
    differencing; contiguous integers collapse the per-integer spans to ``[lo − _HALF, hi + _HALF)``,
    which is what tiles a full ladder without gaps or overlaps. ``open_lo`` / ``open_hi`` mark
    tail buckets (``≤X`` / ``≥Y``): the open end is the ∓∞ sentinel, only the closed end offsets.
    """
    if open_lo:
        if hi is None:
            raise ValueError("integers_in_bucket: open_lo requires a finite upper integer hi.")
        return (-math.inf, float(hi) + _HALF)
    if open_hi:
        if lo is None:
            raise ValueError("integers_in_bucket: open_hi requires a finite lower integer lo.")
        return (float(lo) - _HALF, math.inf)

    if lo is None or hi is None:
        raise ValueError("integers_in_bucket: a closed bucket needs both integer edges lo, hi.")
    if hi < lo:
        raise ValueError(f"integers_in_bucket: inverted bucket (lo={lo} > hi={hi}).")
    return (float(lo) - _HALF, float(hi) + _HALF)


def _normal_cdf_scalar(z: float) -> float:
    """Scalar standard-normal CDF via the one promoted erf source of truth (D-04, Pitfall 6)."""
    return float(normal_cdf(np.array([z], dtype=np.float64))[0])


def bucket_prob(
    mu: float,
    sigma: float,
    lo: float,
    hi: float,
    open_lo: bool = False,
    open_hi: bool = False,
) -> float:
    """Probability mass in one continuous bucket by CDF differencing (D-04).

    ``Φ_blend(hi) − Φ_blend(lo)`` via the erf-based ``normal_cdf`` (Pitfall 6); ``lo``/``hi`` are
    CONTINUOUS edges already offset by ``±_HALF`` (see :func:`integers_in_bucket`). ``open_hi``
    collapses the upper edge to the ``1.0`` tail; ``open_lo`` the lower edge to ``0.0``. Fails
    loud (ASVS V5 / T-04-09): ``mu`` finite, ``sigma`` finite and > 0.
    """
    if not math.isfinite(mu):
        raise ValueError(f"bucket_prob: mu must be finite, got {mu!r}.")
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(f"bucket_prob: sigma must be finite and > 0, got {sigma!r}.")

    upper = 1.0 if open_hi else _normal_cdf_scalar((hi - mu) / sigma)
    lower = 0.0 if open_lo else _normal_cdf_scalar((lo - mu) / sigma)
    return upper - lower


def bucket_probs(
    mu: float,
    sigma: float,
    ladder: Sequence[tuple[float, float, bool, bool]],
) -> NDArray[np.float64]:
    """Probabilities across a full bucket ladder, summing to ~1 (D-04).

    ``ladder`` is a sequence of ``(lo, hi, open_lo, open_hi)`` continuous buckets tiling the
    line (open tails included); returns one float probability per bucket. A gapless tiling of
    ``(−∞, ∞)`` sums to ~1. ``mu``/``sigma`` are guarded per bucket by :func:`bucket_prob`.
    """
    return np.array(
        [bucket_prob(mu, sigma, lo, hi, open_lo, open_hi) for lo, hi, open_lo, open_hi in ladder],
        dtype=np.float64,
    )


# Kalshi ``KXHIGH{CITY}`` ticker suffix → registry city key (A10): the suffix is Kalshi's
# market code, NOT the internal registry key (``NY`` ≠ ``NYC``). Suffixes are LOW-confidence.
TICKER_CITY_SUFFIX_TO_KEY: dict[str, str] = {
    "NY": "NYC",
    "CHI": "CHI",
    "MIA": "MIA",
    "LAX": "LAX",
    "DEN": "DEN",
    "PHIL": "PHI",
    "AUS": "AUS",
}

# ``KXHIGH<SUFFIX>-<lo>-<hi>`` closed-range ticker (e.g. ``KXHIGHNY-62-63``); open-tail
# tickers are parsed from the label form.
_TICKER_RANGE_RE = re.compile(r"^KXHIGH(?P<suffix>[A-Z]+)-(?P<lo>-?\d+)-(?P<hi>-?\d+)$")

# Human-label forms on ``yes_sub_title`` / ``subtitle``; the ``°`` is optional so unicode and
# ASCII forms both round-trip.
_LABEL_RANGE_RE = re.compile(r"^\s*(?P<lo>-?\d+)\s*°?\s*to\s*(?P<hi>-?\d+)\s*°?\s*$", re.IGNORECASE)
_LABEL_LE_RE = re.compile(r"^\s*(?:≤|<=)\s*(?P<v>-?\d+)\s*°?\s*$")
_LABEL_GE_RE = re.compile(r"^\s*(?:≥|>=)\s*(?P<v>-?\d+)\s*°?\s*$")
_LABEL_BELOW_RE = re.compile(r"^\s*(?P<v>-?\d+)\s*°?\s*or\s*below\s*$", re.IGNORECASE)
_LABEL_ABOVE_RE = re.compile(r"^\s*(?P<v>-?\d+)\s*°?\s*or\s*above\s*$", re.IGNORECASE)

# Structured ``strike_type`` values from the Kalshi market record (∈ {between, greater, less, ...}).
_STRIKE_TYPE_LESS = {"less", "less_or_equal", "below"}
_STRIKE_TYPE_GREATER = {"greater", "greater_or_equal", "above"}
_STRIKE_TYPE_BETWEEN = {"between", "range", "in_range"}


def parse_ticker(
    ticker: str | None = None,
    *,
    floor_strike: int | None = None,
    cap_strike: int | None = None,
    strike_type: str | None = None,
    label: str | None = None,
) -> tuple[int | None, int | None, bool, bool]:
    """Pure ``ticker/strike → (lo, hi, open_lo, open_hi)`` edge parser, fail-loud (D-06).

    PURE (no I/O). Returns inclusive integer-degree bounds plus open-tail markers (open-low
    ``≤X`` ⇒ ``lo=None, open_lo=True``; open-high ``≥Y`` ⇒ ``hi=None, open_hi=True``).
    Precedence: structured strikes (``floor_strike``/``cap_strike`` + ``strike_type``, the
    authoritative path) > positional ``ticker`` > ``label``. Fails LOUD (ASVS V5 / T-04-07):
    empty, non-numeric, unrecognized, or inverted input raises rather than default an edge.
    """
    # 1. Structured strikes win (authoritative).
    if strike_type is not None or floor_strike is not None or cap_strike is not None:
        return _parse_structured(floor_strike, cap_strike, strike_type)

    # 2. Positional KXHIGH ticker.
    if ticker is not None:
        return _parse_ticker_string(ticker)

    # 3. Human label fallback.
    if label is not None:
        return _parse_label(label)

    raise ValueError(
        "parse_ticker: no input — supply a ticker, structured strikes, or a label."
    )


def _closed_range(lo: int, hi: int) -> tuple[int, int, bool, bool]:
    """Validate and return a closed ``(lo, hi, False, False)`` bucket, failing loud if inverted."""
    if hi < lo:
        raise ValueError(f"parse_ticker: inverted bucket (lo={lo} > hi={hi}).")
    return (lo, hi, False, False)


def _parse_structured(
    floor_strike: int | None,
    cap_strike: int | None,
    strike_type: str | None,
) -> tuple[int | None, int | None, bool, bool]:
    """Parse the structured Kalshi strike fields (preferred path)."""
    st = strike_type.strip().lower() if strike_type is not None else None

    if st in _STRIKE_TYPE_LESS:
        if cap_strike is None:
            raise ValueError("parse_ticker: a 'less' strike needs cap_strike.")
        return (None, int(cap_strike), True, False)
    if st in _STRIKE_TYPE_GREATER:
        if floor_strike is None:
            raise ValueError("parse_ticker: a 'greater' strike needs floor_strike.")
        return (int(floor_strike), None, False, True)
    if st in _STRIKE_TYPE_BETWEEN or st is None:
        if floor_strike is None or cap_strike is None:
            raise ValueError(
                "parse_ticker: a closed (between) strike needs both floor_strike and cap_strike."
            )
        return _closed_range(int(floor_strike), int(cap_strike))

    raise ValueError(f"parse_ticker: unrecognized strike_type {strike_type!r}.")


def _parse_ticker_string(ticker: str) -> tuple[int | None, int | None, bool, bool]:
    """Parse a ``KXHIGH{SUFFIX}-lo-hi`` closed-range ticker (fail loud on anything else)."""
    if not ticker or not ticker.strip():
        raise ValueError("parse_ticker: empty ticker.")
    m = _TICKER_RANGE_RE.match(ticker.strip())
    if m is None:
        raise ValueError(f"parse_ticker: unrecognized ticker {ticker!r}.")
    suffix = m.group("suffix")
    if suffix not in TICKER_CITY_SUFFIX_TO_KEY:
        raise ValueError(f"parse_ticker: unknown KXHIGH city suffix {suffix!r}.")
    return _closed_range(int(m.group("lo")), int(m.group("hi")))


def _parse_label(label: str) -> tuple[int | None, int | None, bool, bool]:
    """Parse a human ``subtitle``/``yes_sub_title`` label (fallback path, fail loud)."""
    if not label or not label.strip():
        raise ValueError("parse_ticker: empty label.")
    text = label.strip()

    m = _LABEL_RANGE_RE.match(text)
    if m is not None:
        return _closed_range(int(m.group("lo")), int(m.group("hi")))
    m = _LABEL_LE_RE.match(text) or _LABEL_BELOW_RE.match(text)
    if m is not None:
        return (None, int(m.group("v")), True, False)
    m = _LABEL_GE_RE.match(text) or _LABEL_ABOVE_RE.match(text)
    if m is not None:
        return (int(m.group("v")), None, False, True)

    raise ValueError(f"parse_ticker: unrecognized label {label!r}.")
