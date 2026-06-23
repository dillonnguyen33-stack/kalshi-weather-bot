"""D-16 KXHIGH money-path cross-check — live-gated, records-the-gap, fail-loud (TEST-ONLY).

This closes the carried-over Phase-4 verification debt (D-16) by cross-checking the three
deferred KXHIGH money-path facts against a REAL live (or demo) KXHIGH market and FAILING LOUD
on any contradiction:

  (i)   bucket-edge inclusive-integer / half-degree convention — the encoded
        ``[lo − _HALF, hi + _HALF)`` span (``price.buckets._HALF`` / ``integers_in_bucket``)
        must match the live market's ``floor_strike`` / ``cap_strike`` integers and
        ``strike_type`` (T-04-17);
  (ii)  the KXHIGH taker fee coefficient ``0.07`` vs ``0.035`` (``price.fee.FEE_COEFF``) —
        cross-checked against a live KXHIGH fee field/receipt when the market exposes one
        (T-04-18);
  (iii) ticker-suffix → registry-key coverage — every live KXHIGH ticker suffix must resolve
        through ``parse_ticker`` + the registry, with NO silent unknown-suffix default
        (RESEARCH A2/A10).

TEST-ONLY discipline (threat T-05-21): this module READS the encoded constants and asserts
them against the live market. It NEVER mutates ``price/buckets.py`` or ``price/fee.py`` — any
correction to a money-path constant happens ONLY through the 05-05 Task 2 blocking
human-verify checkpoint, never here.

Credential gating (mirrors the ``pg_engine`` ``DATABASE_URL`` skip in ``tests/conftest.py``):
when ``KALSHI_KEY_ID`` / ``KALSHI_PRIVATE_KEY_PATH`` are unset the live assertions SKIP
cleanly so the fast subset stays green, but the principled-default SANITY assertions still
run (the encoded constants are fail-loud-shaped: ``FEE_COEFF == 0.07`` and a representative
KXHIGH-shaped ladder tiles to ~1 on the encoded ``_HALF`` convention). The skip RECORDS the
gap — it never silently CLAIMS the constants are live-confirmed (the executor carries the gap
forward in the SUMMARY, and the Task 2 checkpoint is where a human locks/corrects/records it).

A live contradiction FAILS LOUD here; Task 2's checkpoint then decides whether/how to correct
the constant — this test never edits the ``price/`` constants itself.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from weatherquant.price.buckets import (
    bucket_probs,
    integers_in_bucket,
)
from weatherquant.price.ticker import TICKER_CITY_SUFFIX_TO_KEY, parse_ticker
from weatherquant.price.fee import FEE_COEFF, exact_fee
from weatherquant.registry import CITIES

# --- Credential gate (no creds -> skip the live half, keep the default-sanity half) --------


def _live_creds() -> tuple[str | None, str | None]:
    """Return ``(KALSHI_KEY_ID, KALSHI_PRIVATE_KEY_PATH)`` or blanks → None (genuinely unset)."""
    key_id = os.environ.get("KALSHI_KEY_ID") or None
    key_path = os.environ.get("KALSHI_PRIVATE_KEY_PATH") or None
    return key_id, key_path


def _have_live_creds() -> bool:
    key_id, key_path = _live_creds()
    return bool(key_id and key_path)


_skip_no_creds = pytest.mark.skipif(
    not _have_live_creds(),
    reason=(
        "KALSHI_KEY_ID / KALSHI_PRIVATE_KEY_PATH unset — skipping the LIVE KXHIGH cross-check "
        "(D-16). The principled fail-loud defaults are retained and the gap is RECORDED (not "
        "silently confirmed). Set both + EXECUTION_MODE=paper to run the live half against the "
        "demo host."
    ),
)

# A real KXHIGH market ticker to cross-check, overridable for the live capture session. The
# default is a representative shape; the operator points it at a currently-open market via env.
_LIVE_KXHIGH_TICKER = os.environ.get("KXHIGH_CROSSCHECK_TICKER", "KXHIGHNY")

# The demo REST host is the safe default for the live capture (a real-money host is never the
# test default); flip to prod only if the operator explicitly sets KXHIGH_CROSSCHECK_PROD=1.
_USE_PROD_HOST = os.environ.get("KXHIGH_CROSSCHECK_PROD") == "1"


# --- (always-run) principled-default SANITY: the encoded constants are fail-loud ------------
#
# These run with OR without creds. Without creds they are the records-the-gap fallback: they
# assert the encoded defaults are present and well-shaped, but they NEVER claim those defaults
# were confirmed against a live market (that confirmation is the live half + the Task 2 gate).


def test_fee_coeff_default_is_007_and_not_035_fail_loud():
    """The encoded KXHIGH taker fee coefficient is the principled 0.07, never 0.035 (T-04-18).

    Records-the-gap default: without a live fee receipt this asserts the encoded constant is
    the principled 0.07 (and is observably distinct from the alternate 0.035), so a silent
    regression to 0.035 fails loud. It does NOT claim 0.07 is live-confirmed — that is the
    live half + the Task 2 checkpoint.
    """
    assert FEE_COEFF == 0.07
    assert FEE_COEFF != 0.035
    # The two coefficients produce observably different fees on the same order, so a wrong
    # coefficient can never hide behind cent-rounding on a representative order.
    assert exact_fee(100, 0.50, 0.07) != exact_fee(100, 0.50, 0.035)


def test_half_degree_bucket_edge_default_tiles_to_one_fail_loud():
    """The encoded ``[lo − _HALF, hi + _HALF)`` ladder tiles a KXHIGH-shaped ladder to ~1 (T-04-17).

    Records-the-gap default: a representative KXHIGH integer ladder (open low tail, a run of
    closed degrees around the mode, open high tail) built through the encoded
    ``integers_in_bucket`` convention must sum to ~1 and put the modal bucket on μ. A
    half-degree-shifted convention would not tile, so this fails loud on a silent edge bug. It
    does NOT claim the convention is live-confirmed.
    """
    mu, sigma = 72.0, 4.0
    lo_tail, hi_tail = 64, 80  # open ≤64 ... closed 65..79 ... open ≥80
    ladder: list[tuple[float, float, bool, bool]] = []
    ladder.append((*integers_in_bucket(None, lo_tail, open_lo=True), True, False))
    for k in range(lo_tail + 1, hi_tail):
        ladder.append((*integers_in_bucket(k, k), False, False))
    ladder.append((*integers_in_bucket(hi_tail, None, open_hi=True), False, True))

    probs = bucket_probs(mu, sigma, ladder)
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    # The modal CLOSED bucket sits on μ (Pitfall 1's one-degree shift is absent): closed
    # buckets are ladder[1:-1], degrees 65..79; the max-mass closed bucket is degree 72.
    closed_probs = probs[1:-1]
    modal_degree = (lo_tail + 1) + int(np.argmax(closed_probs))
    assert modal_degree == round(mu)


def test_every_registry_suffix_resolves_and_unknown_fails_loud():
    """Every encoded KXHIGH suffix resolves to a registry key; an unknown suffix fails loud (A2/A10).

    Records-the-gap default: the encoded ``TICKER_CITY_SUFFIX_TO_KEY`` map must point every
    suffix at a real registry city (no dangling key), and ``parse_ticker`` must REJECT an
    unknown suffix rather than silently default it to a wrong city. The live half additionally
    confirms the suffixes a real market actually uses are covered.
    """
    assert TICKER_CITY_SUFFIX_TO_KEY, "the suffix→key map must not be empty"
    for suffix, key in TICKER_CITY_SUFFIX_TO_KEY.items():
        assert key in CITIES, f"suffix {suffix!r} maps to unknown registry key {key!r}"
        # A known suffix parses a closed range without raising.
        lo, hi, open_lo, open_hi = parse_ticker(f"KXHIGH{suffix}-62-63")
        assert (lo, hi, open_lo, open_hi) == (62, 63, False, False)

    # An unknown suffix must FAIL LOUD (never silently default — ASVS V5, T-05-18).
    with pytest.raises(ValueError):
        parse_ticker("KXHIGHZZZ-62-63")


# --- (live-gated) the REAL KXHIGH cross-check: fail loud on any contradiction ---------------


@pytest.fixture(scope="module")
def live_kxhigh_market() -> dict:
    """Fetch a real KXHIGH market record via a signed REST ``GetMarket`` (reuses the signer).

    Skipped (never errored) when creds are unset. Reuses ``market.auth.KalshiSigner`` and an
    ``httpx.AsyncClient`` (the same signer + REST seam the WS client/snapshot use) — no
    re-derived auth. The host defaults to DEMO (a real-money host is never the test default).

    Creds may arrive via the shell OR via the ``.env`` ``conftest`` loads — so on a dev box the
    gate has creds even with nothing exported. The default ticker is only a representative
    placeholder (dated KXHIGH tickers expire), so a ``404`` means "no open market to check" —
    that is the records-the-gap SKIP, not a money-path contradiction. Auth/5xx still fail loud.
    """
    if not _have_live_creds():
        pytest.skip("no live creds — handled by _skip_no_creds on the live tests")

    import asyncio

    import httpx

    from weatherquant.db.engine import get_settings
    from weatherquant.market.auth import KalshiSigner
    from weatherquant.market.client import REST_HOST_DEMO, REST_HOST_PROD

    signer = KalshiSigner.from_settings(get_settings())
    rest_host = REST_HOST_PROD if _USE_PROD_HOST else REST_HOST_DEMO
    path = f"/trade-api/v2/markets/{_LIVE_KXHIGH_TICKER}"

    async def _get_market() -> dict:
        async with httpx.AsyncClient() as http:
            headers = dict(signer.sign("GET", path))  # query-stripped path (Pitfall 6)
            response = await http.get(f"{rest_host}{path}", headers=headers)
            if response.status_code == 404:
                # Ticker not an open market (the default is a placeholder; dated tickers
                # expire) — records-the-gap SKIP, not a contradiction. Auth/5xx fall through
                # to raise_for_status and fail loud (a real signing/host problem).
                pytest.skip(
                    f"KXHIGH market {_LIVE_KXHIGH_TICKER!r} returned 404 on the "
                    f"{'PROD' if _USE_PROD_HOST else 'DEMO'} host — no open market to "
                    "cross-check. Set KXHIGH_CROSSCHECK_TICKER to a currently-open KXHIGH "
                    "market to run the live half (D-16); encoded defaults retained, gap recorded."
                )
            response.raise_for_status()
            payload = response.json()
            # The GetMarket response wraps the record under "market".
            return payload.get("market", payload)

    return asyncio.run(_get_market())


@_skip_no_creds
def test_live_bucket_edge_matches_encoded_half_degree_convention(live_kxhigh_market):
    """(i) live ``floor_strike``/``cap_strike`` map to the encoded ``[lo − _HALF, hi + _HALF)``.

    FAILS LOUD (T-04-17) if the live closed (``between``) market's integer strikes do not map
    to the encoded continuous span the money path differences over. The encoded constant is
    READ (``integers_in_bucket``), never mutated — a contradiction is the Task 2 checkpoint's
    to correct.
    """
    market = live_kxhigh_market
    floor_strike = market.get("floor_strike")
    cap_strike = market.get("cap_strike")
    strike_type = market.get("strike_type")

    lo, hi, open_lo, open_hi = parse_ticker(
        ticker=market.get("ticker"),
        floor_strike=floor_strike,
        cap_strike=cap_strike,
        strike_type=strike_type,
    )
    c_lo, c_hi = integers_in_bucket(lo, hi, open_lo=open_lo, open_hi=open_hi)

    if not open_lo:
        # The lower continuous edge is exactly the lo integer minus the encoded half-degree.
        assert c_lo == float(lo) - 0.5, (
            f"live floor_strike={floor_strike} contradicts encoded [lo-0.5, ...) "
            f"(got c_lo={c_lo}). FAIL LOUD (T-04-17) — Task 2 checkpoint decides the fix."
        )
    else:
        assert math.isinf(c_lo) and c_lo < 0
    if not open_hi:
        assert c_hi == float(hi) + 0.5, (
            f"live cap_strike={cap_strike} contradicts encoded [..., hi+0.5) "
            f"(got c_hi={c_hi}). FAIL LOUD (T-04-17) — Task 2 checkpoint decides the fix."
        )
    else:
        assert math.isinf(c_hi) and c_hi > 0


@_skip_no_creds
def test_live_fee_coefficient_matches_encoded_007(live_kxhigh_market):
    """(ii) a live KXHIGH fee field/receipt implies the encoded ``0.07`` coefficient (not 0.035).

    FAILS LOUD (T-04-18) if the live market exposes a taker-fee field/receipt that implies a
    coefficient distinct from the encoded ``FEE_COEFF`` (within cent tolerance). When the
    market record carries no usable fee field this asserts the principled default and records
    the gap (the live receipt cross-check is the Task 2 checkpoint's to confirm). The encoded
    coefficient is READ, never mutated.
    """
    market = live_kxhigh_market

    # Kalshi market records may expose the fee coefficient under a few known keys. Probe them
    # without inventing a value — absence = absence (records-the-gap), presence = fail-loud
    # cross-check.
    fee_coeff_field = None
    for key in ("taker_fee_coefficient", "fee_coefficient", "trading_fee_coefficient"):
        if market.get(key) is not None:
            fee_coeff_field = float(market[key])
            break

    if fee_coeff_field is not None:
        assert fee_coeff_field == pytest.approx(FEE_COEFF, abs=1e-6), (
            f"live KXHIGH fee coefficient {fee_coeff_field} contradicts encoded "
            f"FEE_COEFF={FEE_COEFF}. FAIL LOUD (T-04-18) — Task 2 checkpoint decides the fix."
        )
    else:
        # No machine-readable coefficient on the record: retain the principled default and
        # leave the receipt cross-check to the Task 2 human-verify step (records the gap).
        assert FEE_COEFF == 0.07
        pytest.skip(
            "live KXHIGH market record exposes no fee-coefficient field — the receipt "
            "cross-check (T-04-18) is the Task 2 human-verify step; default 0.07 retained, "
            "gap recorded."
        )


@_skip_no_creds
def test_live_ticker_suffix_resolves_to_registry_key(live_kxhigh_market):
    """(iii) the live KXHIGH ticker's suffix resolves to a registry key (no silent default, A2/A10).

    FAILS LOUD (T-05-18) if the live market's ticker carries a KXHIGH city suffix that
    ``parse_ticker`` cannot resolve through ``TICKER_CITY_SUFFIX_TO_KEY`` → registry — an
    unknown suffix must raise, never silently price the wrong city.
    """
    ticker = live_kxhigh_market.get("ticker") or _LIVE_KXHIGH_TICKER
    # Extract the alphabetic suffix after the KXHIGH prefix (e.g. KXHIGHNY-... → "NY").
    assert ticker.startswith("KXHIGH"), f"unexpected non-KXHIGH ticker {ticker!r}"
    rest = ticker[len("KXHIGH"):]
    suffix = "".join(ch for ch in rest.split("-", 1)[0] if ch.isalpha())

    assert suffix in TICKER_CITY_SUFFIX_TO_KEY, (
        f"live KXHIGH ticker {ticker!r} suffix {suffix!r} resolves to NO registry key "
        f"(known: {sorted(TICKER_CITY_SUFFIX_TO_KEY)}). FAIL LOUD (T-05-18, A2/A10) — record "
        f"the unresolved suffix; the Task 2 checkpoint decides coverage."
    )
    assert TICKER_CITY_SUFFIX_TO_KEY[suffix] in CITIES
