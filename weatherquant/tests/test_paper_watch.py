"""``paper --watch`` feed-driven money path — mock-WS end-to-end (PAP-01/PAP-03, 05.1-02).

The watch loop drives the EXISTING, tested ``market.client.run_feed`` (signed handshake +
reconnect + seq-gap recovery) into the EXISTING money path: an ``on_book`` sink derives the
midpoint off the LIVE ``OrderBook``, runs the shared ``_process_book`` body (EV/Kelly gate +
``taker_sweep``), and persists ``market_snapshots`` stamped with ``OrderBook.event_time`` (the
real WS instant, NEVER now()) at cadence + on material book moves, terminating at the
settlement-window end bounded by ``--max-duration``.

Exercised with a MOCK WS connector (no live network, no creds) reusing the ``_MockWS`` /
``_MockConnector`` harness + the enveloped ``_ws_snapshot`` / ``_ws_delta`` builders from
``test_market_client.py``, and the offline DB/settings/signer/persist patch idiom from
``test_cli.py``'s ``_patch_paper`` (which supplies the patched signer). Each test asserts a
distinct contract:

1. snapshots persist across a closing window stamped with the WS event time (D-08);
2. cadence / material-move density holds so the CLV window stays dense (PAP-04, T-05-20);
3. ``execution_mode == "live"`` bows out BEFORE any feed opens (no order path, T-051-08);
4. CLV is derivable end-to-end over the persisted rows (CONTEXT specifics);
5. a raising persist sink surfaces as ``SystemExit`` after the feed drains (WR-04 / T-051-09).
"""

from __future__ import annotations

from datetime import datetime, timedelta

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


def _run_watch(connector: _MockConnector, *, max_duration: int = 600) -> dict:
    """Drive ``run_paper`` under ``--watch`` with a bounded mock connector (no live network)."""
    args = _paper_args(ticker=_TICKER, watch=True, max_duration=max_duration)
    args.ws_connect = connector
    args.max_reconnects = 1  # bounded: the scripted stream drains, then run_feed returns
    return cli.run_paper(args)


def test_watch_loop_cadence_material_move_density(monkeypatch: pytest.MonkeyPatch):
    """PAP-04 / T-05-20: in-window material book moves keep the CLV closing window dense.

    Two in-window deltas, each MOVING the top-of-book (a yes-bid size cut, then a no-bid add that
    shifts the reflected yes ask), persist within the same debounce interval BECAUSE each is a
    material move — so the closing window holds >= 1 persisted snapshot even though the deltas land
    seconds apart (finer than PAPER_SNAPSHOT_CADENCE_SECONDS). Scripted ts values keep it
    deterministic (no real time).
    """
    captured = _patch_watch(monkeypatch)
    ts1 = _in_window_instant(8)
    ts2 = _in_window_instant(7)  # ~1 min later, well inside the debounce interval
    conn = _MockWS(
        [
            _ws_snapshot(1, ticker=_TICKER),
            # A yes-bid size cut (top-of-book yes size moves) — a material move.
            _ws_delta(2, side="yes", price="0.40", delta="-50.00", ts=_iso(ts1), ticker=_TICKER),
            # A no-bid add at a NEW best no price (reflected yes ask moves) — a material move.
            _ws_delta(3, side="no", price="0.57", delta="100.00", ts=_iso(ts2), ticker=_TICKER),
        ],
        close_after=False,
    )
    _run_watch(_MockConnector([conn]))

    win = settlement_window(get_city("NYC"), _PAPER_DATE)
    window_start = win.end_utc - timedelta(minutes=clv.CLV_WINDOW_MINUTES)
    in_window = [
        kw for kw in captured["snapshots"] if window_start <= kw["available_at"] < win.end_utc
    ]
    assert len(in_window) >= 1  # the window is dense (PAP-04)
    # A top-of-book move forced a second persist inside the debounce interval (material-move path).
    assert len(captured["snapshots"]) >= 2
    assert cli.PAPER_SNAPSHOT_CADENCE_SECONDS < clv.CLV_WINDOW_MINUTES * 60


def test_watch_bows_out_of_live_mode_before_any_feed(monkeypatch: pytest.MonkeyPatch):
    """T-051-08: execution_mode=='live' raises SystemExit BEFORE any feed is opened (no order path).

    The mock connector records every (re)connection; with the live bow-out running first its
    ``connect_calls`` stays EMPTY — no WS handshake, no order path, the structural guard holds.
    """
    _patch_watch(monkeypatch)
    # Flip the patched settings to live (the bow-out is the FIRST thing run_paper checks).
    monkeypatch.setattr(
        cli.paper,
        "get_settings",
        lambda: type("S", (), {"max_position_fraction": 0.025, "execution_mode": "live"})(),
    )
    conn = _MockWS([_ws_snapshot(1, ticker=_TICKER)], close_after=False)
    connector = _MockConnector([conn])

    args = _paper_args(ticker=_TICKER, watch=True, max_duration=600)
    args.ws_connect = connector
    args.max_reconnects = 1
    with pytest.raises(SystemExit, match="execution_mode='live'"):
        cli.run_paper(args)

    # The feed was NEVER opened — the bow-out ran before any connection (no order-submission path).
    assert connector.connect_calls == []


def test_watch_clv_derivable_end_to_end(monkeypatch: pytest.MonkeyPatch):
    """CONTEXT specifics: persisted watch snapshots yield a NON-EMPTY closing window + a finite mid.

    Drive a scripted closing-window stream, then feed the persisted rows (captured offline) back
    through clv.closing_window_snapshots + clv.vol_weighted_mid and assert a non-empty window and a
    derivable (finite, positive-cents) closing mid — the end-to-end CLV-derivable contract.
    """
    captured = _patch_watch(monkeypatch)
    ts1 = _in_window_instant(10)
    ts2 = _in_window_instant(5)
    conn = _MockWS(
        [
            _ws_snapshot(1, ticker=_TICKER),
            _ws_delta(2, side="yes", price="0.40", delta="-30.00", ts=_iso(ts1), ticker=_TICKER),
            _ws_delta(3, side="no", price="0.57", delta="50.00", ts=_iso(ts2), ticker=_TICKER),
        ],
        close_after=False,
    )
    _run_watch(_MockConnector([conn]))

    # The captured persist kwargs ARE market_snapshots-shaped rows (available_at + mid + volume).
    rows = captured["snapshots"]
    assert rows, "the watch loop persisted no snapshot to derive CLV from"
    window = clv.closing_window_snapshots(rows, get_city("NYC"), _PAPER_DATE)
    assert window, "the persisted rows produced an EMPTY closing window (CLV not derivable)"
    closing_mid = clv.vol_weighted_mid(window)
    assert closing_mid == closing_mid  # finite (not NaN)
    assert 0.0 < closing_mid < 100.0  # a derivable cents mid in the valid band


def test_watch_persist_failure_surfaces_as_systemexit(monkeypatch: pytest.MonkeyPatch):
    """WR-04 / T-051-09: a raising persist sink surfaces as SystemExit AFTER the feed drains.

    client._emit CATCHES the sink's raise so the feed keeps running (a hiccup must not kill the
    long-running feed). A HARD persist failure must still fail loud — the loop inspects the sink's
    recorded error after the feed drains and re-raises SystemExit, so a silently-swallowed DB error
    can never make --watch look successful. Offline (no DB, mock connector).
    """
    _patch_watch(monkeypatch)

    def _raising_persist_snapshot(bind, **kw):  # noqa: ANN001
        raise RuntimeError("simulated DB write failure")

    monkeypatch.setattr(cli.paper, "persist_snapshot", _raising_persist_snapshot)

    ts = _in_window_instant(5)
    conn = _MockWS(
        [_ws_snapshot(1, ticker=_TICKER), _ws_delta(2, ts=_iso(ts), ticker=_TICKER)],
        close_after=False,
    )
    with pytest.raises(SystemExit, match="persist failed"):
        _run_watch(_MockConnector([conn]))
