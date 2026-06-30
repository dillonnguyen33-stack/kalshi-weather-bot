"""kxhigh-ticker money-path regression — real date-coded tickers price/settle via the structured path.

The bug (latent since Phase 5): the live money path (``run_price`` / ``run_paper`` → ``_price_bucket``
→ ``price.parse_ticker``, and ``cli.verify._settle_window_fills``) resolved bucket edges ONLY from
the ticker STRING via the positional ``^KXHIGH(SUFFIX)-lo-hi$`` regex, which CANNOT match a real
date-coded Kalshi ticker (``KXHIGHNY-26JUN18-T72``, ``KXHIGHAUS-26JUN23-B93.5``) — so every real
fill raised ``unrecognized ticker`` and crashed pricing + settlement. It was INVISIBLE because the
suite only ever used real tickers as OPAQUE KEYS, never parsing them to edges.

Option B fix (fetch + persist structured strikes), proven here OFFLINE / DB-free:

1. ``market.client.fetch_market`` signs a ``GetMarket`` read and returns the structured
   ``floor_strike`` / ``cap_strike`` / ``strike_type`` the live-confirmed ``parse_ticker`` path
   consumes (never reverse-engineering the B/T ticker-string grammar);
2. ``_resolve_bucket`` resolves a real date-coded ticker via those structured fields (where the
   positional parse RAISES), failing loud on a demo-scaled ``8.5e-05`` strike (never int()-zero);
3. ``run_price`` / ``run_paper`` drive the structured path for a real ticker and PERSIST the resolved
   edges into each fill's ``detail['bucket']``;
4. ``cli.verify._settle_window_fills`` settles OFFLINE from the persisted edges, failing loud when
   neither a persisted bucket nor a parseable ticker is present.

Mirrors ``tests/test_paper_watch.py`` + the mock-WS/HTTP harness from ``test_market_client.py`` /
``test_cli.py`` (no live network, no creds, no DB).
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from tests.test_cli import (
    _PAPER_DATE,
    _cal_row,
    _forecast_rows,
    _paper_args,
    _patch_paper,
    _patch_price_db,
    _price_args,
)
from tests.test_market_client import _MockHttp
from weatherquant import cli
from weatherquant.market.client import REST_HOST_PROD, fetch_market
from weatherquant.price.ticker import parse_ticker
from weatherquant.registry import get_city
from weatherquant.time import settlement_window

# Real date-coded Kalshi KXHIGH tickers the positional ``KXHIGH{SUFFIX}-lo-hi`` regex CANNOT match
# (the exact shapes flagged in 05.1-UAT Test 2 NOTE). The B/T tail direction is NOT decoded here —
# the structured ``strike_type`` from the mocked GetMarket record drives resolution, never a guess.
_REAL_THRESHOLD_TICKER = "KXHIGHNY-26JUN18-T72"
_REAL_OPEN_LOW_TICKER = "KXHIGHAUS-26JUN23-B93.5"


class _StubSigner:
    """Offline KalshiSigner stand-in (no key material) for the structured-fetch resolution path."""

    @classmethod
    def from_settings(cls, settings):  # noqa: ANN001
        return cls()

    def sign(self, method, path):  # noqa: ANN001
        return {}


# --- market.client.fetch_market: the signed GetMarket structured-strike read (HTTP-level) --------


async def test_fetch_market_returns_structured_strikes_query_stripped():
    """fetch_market signs the GetMarket path query-stripped and returns the structured strikes.

    Mirrors test_fetch_snapshot_signs_query_stripped_path_*: the record wraps under ``"market"``
    and the path carries NO query (Pitfall 6 / T-05-06); the host stays a fixed const (T-05-11).
    """
    signed: list[tuple[str, str]] = []

    def recording_signer(method, path):  # noqa: ANN001
        signed.append((method, path))
        return {"KALSHI-ACCESS-KEY": "k"}

    payload = {
        "market": {
            "ticker": _REAL_THRESHOLD_TICKER,
            "floor_strike": 72,
            "cap_strike": None,
            "strike_type": "greater",
        }
    }
    rec = await fetch_market(_MockHttp(payload), recording_signer, _REAL_THRESHOLD_TICKER)

    assert signed == [("GET", f"/trade-api/v2/markets/{_REAL_THRESHOLD_TICKER}")]
    assert "?" not in signed[0][1]
    assert rec == {
        "ticker": _REAL_THRESHOLD_TICKER,
        "floor_strike": 72,
        "cap_strike": None,
        "strike_type": "greater",
    }


async def test_fetch_market_404_fails_loud_no_such_market():
    """A 404 surfaces a clear "no such market" ValueError BEFORE raise_for_status (dated tickers expire)."""

    class _Resp:
        status_code = 404
        headers: dict = {}

        def raise_for_status(self):
            raise AssertionError("a 404 must short-circuit before raise_for_status")

        def json(self):
            return {}

    class _Http:
        async def get(self, url, *, params=None, headers=None):  # noqa: ANN001
            return _Resp()

    with pytest.raises(ValueError, match="no such market"):
        await fetch_market(_Http(), lambda m, p: {}, _REAL_THRESHOLD_TICKER)


async def test_fetch_market_missing_market_record_fails_loud():
    """A GetMarket payload with no ``market`` record fails loud (never a fabricated bucket)."""
    http = _MockHttp({"not_market": {"floor_strike": 72}})
    with pytest.raises(ValueError, match="no 'market' record"):
        await fetch_market(http, lambda m, p: {}, _REAL_THRESHOLD_TICKER)


# --- the bug + the structured-resolution fix (pure, offline) -------------------------------------


def test_positional_parse_rejects_real_date_coded_tickers():
    """THE BUG: the positional regex CANNOT match a real date-coded KXHIGH ticker (it raises)."""
    for ticker in (_REAL_THRESHOLD_TICKER, _REAL_OPEN_LOW_TICKER):
        with pytest.raises(ValueError, match="unrecognized ticker"):
            parse_ticker(ticker)
        # ...so the money path knows it must fetch the structured GetMarket record for this ticker.
        assert cli.pricing._needs_market_record(ticker) is True


def test_synthetic_positional_ticker_needs_no_market_record():
    """The synthetic ``KXHIGH{SUFFIX}-lo-hi`` form parses offline — no structured fetch needed."""
    assert cli.pricing._needs_market_record("KXHIGHNY-62-63") is False


def test_structured_strikes_resolve_real_threshold_ticker():
    """A real threshold ticker resolves via the structured greater-strike → open-high tail."""
    bucket = cli.pricing._resolve_bucket(
        _REAL_THRESHOLD_TICKER,
        market={"floor_strike": 72, "cap_strike": None, "strike_type": "greater"},
    )
    assert bucket == (72, None, False, True)


def test_structured_strikes_resolve_real_open_low_ticker():
    """A real open-low ticker resolves via the structured less-strike → open-low tail (cap=93)."""
    bucket = cli.pricing._resolve_bucket(
        _REAL_OPEN_LOW_TICKER,
        market={"floor_strike": None, "cap_strike": 93, "strike_type": "less"},
    )
    assert bucket == (None, 93, True, False)


def test_structured_strikes_resolve_real_between_ticker():
    """A real between ticker resolves via the structured floor+cap strikes → closed bucket."""
    bucket = cli.pricing._resolve_bucket(
        _REAL_OPEN_LOW_TICKER,
        market={"floor_strike": 92, "cap_strike": 93, "strike_type": "between"},
    )
    assert bucket == (92, 93, False, False)


def test_demo_scaled_strike_fails_loud_never_int_zero():
    """A demo-scaled ``8.5e-05`` strike fails LOUD rather than int()-zeroing to bucket edge 0 (D-06).

    The 05.1-UAT demo returns ``8.5e-05`` for 85°F; ``int(8.5e-05) == 0`` would silently price the
    wrong bucket. The scale-sanity guard rejects it (the exact degenerate-strike money-path hazard).
    """
    with pytest.raises(SystemExit, match="scaled"):
        cli.pricing._resolve_bucket(
            _REAL_THRESHOLD_TICKER,
            market={"floor_strike": 8.5e-05, "cap_strike": None, "strike_type": "greater"},
        )


# --- run_price bucket resolution: structured fetch for a real ticker, offline for the synthetic ---


def test_resolve_price_bucket_fetches_structured_for_real_ticker(monkeypatch: pytest.MonkeyPatch):
    """_resolve_price_bucket fetches the GetMarket record for a real ticker and resolves its edges."""
    captured: dict = {}

    def _fake_fetch(ticker, sign, rest_host):  # noqa: ANN001
        captured.update(ticker=ticker, rest_host=rest_host)
        return {"floor_strike": 72, "cap_strike": None, "strike_type": "greater"}

    monkeypatch.setattr(cli.pricing, "_fetch_market_record", _fake_fetch)
    monkeypatch.setattr(cli.pricing, "KalshiSigner", _StubSigner)

    bucket = cli.pricing._resolve_price_bucket(
        _REAL_THRESHOLD_TICKER, settings=object(), demo=False
    )
    assert bucket == (72, None, False, True)
    assert captured["ticker"] == _REAL_THRESHOLD_TICKER
    assert captured["rest_host"] == REST_HOST_PROD  # demo=False → fixed prod host (SSRF guard)


def test_resolve_price_bucket_offline_for_synthetic_ticker(monkeypatch: pytest.MonkeyPatch):
    """A synthetic positional ticker resolves offline — no signer built, no record fetched."""

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("a synthetic positional ticker must NOT fetch a market record")

    monkeypatch.setattr(cli.pricing, "_fetch_market_record", _boom)
    bucket = cli.pricing._resolve_price_bucket("KXHIGHNY-62-63", settings=object(), demo=False)
    assert bucket == (62, 63, False, False)


def test_run_price_prices_real_date_coded_ticker_via_structured_fetch(
    monkeypatch: pytest.MonkeyPatch,
):
    """run_price END-TO-END: a real threshold ticker that USED to crash now prices via structured strikes.

    μ≈73, σ=1 over the open-high ``≥72`` bucket → high YES prob; vs the mocked mid 0.10 the edge
    (and the Kelly stake sized on the same shrunk ``p_used``) is unambiguously positive.
    """
    _patch_price_db(
        monkeypatch, forecasts=_forecast_rows("hrrr", 73.0), cal_rows=[_cal_row("hrrr")]
    )
    monkeypatch.setattr(
        cli.pricing,
        "_fetch_market_record",
        lambda ticker, sign, rest_host: {
            "floor_strike": 72,
            "cap_strike": None,
            "strike_type": "greater",
        },
    )
    monkeypatch.setattr(cli.pricing, "KalshiSigner", _StubSigner)

    result = cli.run_price(_price_args(ticker=_REAL_THRESHOLD_TICKER, market_mid="0.1"))
    bucket = result["buckets"][0]
    assert bucket["ticker"] == _REAL_THRESHOLD_TICKER
    assert bucket["prob"] > 0.5
    assert bucket["ev"] > 0.0
    assert bucket["stake_fraction"] > 0.0


# --- run_paper END-TO-END: the real ticker prices, fills, and PERSISTS the structured bucket ------


def test_run_paper_real_ticker_prices_and_persists_structured_bucket(
    monkeypatch: pytest.MonkeyPatch,
):
    """run_paper END-TO-END on a real date-coded ticker: it prices + fills (no crash) and persists the bucket.

    A scripted two-sided book (yes bid 40¢, no bid 56¢ → yes ask 44¢ → mid 0.42) inside the NYC
    closing window, μ≈73 over the open-high ``≥72`` bucket → positive EV → a taker fill. The bug
    used to raise ``unrecognized ticker`` here; now the resolved structured edges are PERSISTED into
    ``detail['bucket']`` so ``cli.verify`` can settle the fill offline.
    """
    win = settlement_window(get_city("NYC"), _PAPER_DATE)
    event_time = win.end_utc - timedelta(minutes=5)  # inside the CLV window
    book = {
        "type": "orderbook_snapshot",
        "seq": 7,
        "ticker": _REAL_THRESHOLD_TICKER,
        "yes": [[40, 200]],
        "no": [[56, 200]],
        "event_time": event_time,
    }
    captured = _patch_paper(
        monkeypatch,
        book=book,
        forecasts=_forecast_rows("hrrr", 73.0),
        cal_rows=[_cal_row("hrrr")],
    )
    monkeypatch.setattr(
        cli.pricing,
        "_fetch_market_record",
        lambda ticker, sign, rest_host: {
            "floor_strike": 72,
            "cap_strike": None,
            "strike_type": "greater",
        },
    )

    result = cli.run_paper(_paper_args(ticker=_REAL_THRESHOLD_TICKER))

    assert result["fill"] is not None  # priced + filled (used to crash on parse_ticker)
    fill_kw = captured["fills"][0]
    assert fill_kw["detail"]["bucket"] == {
        "lo": 72,
        "hi": None,
        "open_lo": False,
        "open_hi": True,
    }


# --- cli.verify._settle_window_fills: settle OFFLINE from the persisted edges --------------------


def test_settlement_reads_persisted_bucket_edges_for_real_ticker():
    """Settlement reads the persisted ``detail['bucket']`` edges — no re-parse of the real ticker string."""
    fill = {
        "ticker": _REAL_THRESHOLD_TICKER,
        "detail": {"bucket": {"lo": 72, "hi": None, "open_lo": False, "open_hi": True}},
    }
    edges = cli.verify._fill_bucket_edges(fill, _REAL_THRESHOLD_TICKER, parse_ticker)
    assert edges == (72, None, False, True)


def test_settlement_falls_back_to_positional_for_synthetic_fill():
    """A legacy synthetic-form fill with no persisted bucket falls back to the positional parse."""
    fill = {"ticker": "KXHIGHNY-62-63", "detail": None}
    edges = cli.verify._fill_bucket_edges(fill, "KXHIGHNY-62-63", parse_ticker)
    assert edges == (62, 63, False, False)


def test_settlement_fails_loud_on_real_ticker_without_persisted_bucket():
    """A real-ticker fill carrying NO persisted bucket fails loud — never a fabricated bucket."""
    fill = {"ticker": _REAL_THRESHOLD_TICKER, "detail": {"avg_price_cents": 44.0}}
    with pytest.raises(ValueError, match="unrecognized ticker"):
        cli.verify._fill_bucket_edges(fill, _REAL_THRESHOLD_TICKER, parse_ticker)
