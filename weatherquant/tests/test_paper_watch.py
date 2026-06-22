"""``paper --watch`` feed-driven money path — mock-WS end-to-end (PAP-01/PAP-03, 05.1-02).

The watch loop drives the EXISTING, tested ``market.client.run_feed`` (signed handshake +
reconnect + seq-gap recovery) into the EXISTING money path: an ``on_book`` sink derives the
midpoint off the LIVE ``OrderBook``, runs the shared ``_process_book`` body (EV/Kelly gate +
``taker_sweep``), and persists ``market_snapshots`` stamped with ``OrderBook.event_time`` (the
real WS instant, NEVER now()) at cadence + on material book moves, terminating at the
settlement-window end bounded by ``--max-duration``.

Exercised with a MOCK WS connector (no live network, no creds) reusing the ``_MockWS`` /
``_MockConnector`` / ``_signer`` harness + the enveloped ``_ws_snapshot`` / ``_ws_delta``
builders from ``test_market_client.py``, and the offline DB/settings/signer/persist patch idiom
from ``test_cli.py``'s ``_patch_paper``. Each test asserts a distinct contract:

1. snapshots persist across a closing window stamped with the WS event time (D-08);
2. cadence / material-move density holds so the CLV window stays dense (PAP-04, T-05-20);
3. ``execution_mode == "live"`` bows out BEFORE any feed opens (no order path, T-051-08);
4. CLV is derivable end-to-end over the persisted rows (CONTEXT specifics);
5. a raising persist sink surfaces as ``SystemExit`` after the feed drains (WR-04 / T-051-09).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from weatherquant import cli
from weatherquant.market import clv
from weatherquant.registry import get_city
from weatherquant.time import settlement_window

# Reuse the verified mock-WS harness + enveloped builders rather than re-deriving them — the
# watch loop consumes run_feed exactly as test_market_client.py exercises it.
from tests.test_market_client import (
    _MockConnector,
    _MockWS,
    _signer,
    _ws_delta,
    _ws_snapshot,
)
from tests.test_cli import (
    _PAPER_DATE,
    _PAPER_TICKER,
    _cal_row,
    _forecast_rows,
    _paper_args,
    _patch_paper,
)

# These tests call the SYNC ``run_paper`` (which runs its own ``asyncio.run`` for the watch
# loop) — so they are NOT asyncio tests; an outer running loop would clash with that inner
# ``asyncio.run``. The async machinery lives entirely inside run_paper's watch branch.

# Use the RANGE ticker the money path's _price_bucket parses (the WS-harness default
# KXHIGHNY-...-T72 is a threshold ticker the bucket pricer rejects); the mock builders take a
# ``ticker=`` kwarg so the envelope's market_ticker matches.
_TICKER = _PAPER_TICKER


def _iso(ts: datetime) -> str:
    """Render a tz-aware UTC instant as the ``Z``-suffixed ISO string the WS wire carries."""
    return ts.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _in_window_instant(minutes_before_end: int) -> datetime:
    """A tz-aware UTC instant ``minutes_before_end`` before the NYC settlement end (in-window)."""
    win = settlement_window(get_city("NYC"), _PAPER_DATE)
    return win.end_utc - timedelta(minutes=minutes_before_end)


def _patch_watch(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Patch the offline money-path seams for the watch path (book arg unused under --watch)."""
    return _patch_paper(
        monkeypatch,
        book={"type": "orderbook_snapshot", "seq": 1, "ticker": _TICKER, "yes": [], "no": []},
        forecasts=_forecast_rows("hrrr", 62.5),
        cal_rows=[_cal_row("hrrr")],
    )


def test_watch_loop_persists_ws_stamped_snapshot(monkeypatch: pytest.MonkeyPatch):
    """A WS snapshot + an in-window delta persists ONE snapshot stamped with the WS ts (D-08).

    The snapshot (seq=1) carries no event time, so the sink skips it (never now()); the first
    delta (seq=2) carries ``msg.ts`` inside the closing window, so the sink persists a snapshot
    whose ``available_at`` EQUALS that WS instant — proving the WS event time, not the wall
    clock, is the stamp (extends test_event_time_surfaced_to_on_book to the persisted column).
    """
    captured = _patch_watch(monkeypatch)
    ts = _in_window_instant(5)
    conn = _MockWS(
        [_ws_snapshot(1, ticker=_TICKER), _ws_delta(2, ts=_iso(ts), ticker=_TICKER)],
        close_after=False,
    )
    connector = _MockConnector([conn])

    args = _paper_args(ticker=_TICKER, watch=True, max_duration=600)
    args.ws_connect = connector
    args.max_reconnects = 1  # bounded: the scripted stream drains, then run_feed returns
    result = cli.run_paper(args)

    assert result["persisted_snapshot_count"] >= 1
    assert len(captured["snapshots"]) >= 1
    # Every persisted snapshot is stamped with the WS instant, never now().
    assert all(kw["available_at"] == ts for kw in captured["snapshots"])
