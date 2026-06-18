"""Kalshi market edge for the paper-fill simulator (Phase 5).

``weatherquant.market`` is the I/O edge that talks to Kalshi (the live orderbook WS feed,
the REST snapshot, RSA-PSS request signing) and the pure correctness seams the fill
simulator prices against. It is fenced OUT of the pure ``weatherquant.price`` money path by
the ``tests/test_no_market_leak_into_price.py`` AST guard — ``price/`` never imports
``market``/``websockets``/``cryptography``.

Public surface (grows over 05-02..05-04):

* ``auth`` — :class:`KalshiSigner`, :func:`load_key`, :func:`sign`: the ONE RSA-PSS signer
  shared by the WS handshake and the REST snapshot (PAP-01, D-01).
* ``reflect`` — :func:`yes_ask_levels`, :func:`no_ask_levels`: the ONE yes/no bid-only
  reflection seam (``ask = 100 - opposite bid``) every fill price/size routes through
  (PAP-02, the central correctness landmine).
* ``book`` — :class:`OrderBook`, :class:`SeqGap`, :func:`apply`: the in-memory per-ticker
  book + ``seq`` integrity (a gap raises ``SeqGap``, a ``CorrectnessError``, triggering a
  REST re-snapshot — never a silent carry-forward, PAP-01, D-02).
* ``client`` — :func:`run_feed`, :func:`fetch_snapshot`: the signed WS connect + auto-
  reconnect loop (re-subscribe AND REST re-snapshot on every reconnection) and the signed
  REST orderbook snapshot resync anchor (PAP-01).
* ``persist`` — :func:`persist_snapshot`, :func:`persist_fill`, :func:`latest_snapshots`:
  the THIN snapshot/fill write+read adapter over the one audited append-only writer path and
  ``queries.latest`` (no Core insert here — D-13, threat T-05-15).
* ``clv`` — :func:`clv_cents`, :func:`vol_weighted_mid`, :func:`closing_window_snapshots`,
  :data:`CLV_WINDOW_MINUTES`: the PURE derived per-trade CLV against a volume-weighted closing
  mid, anchored on the one LST ``settlement_window`` clock (PAP-04, D-09/D-10/D-12).
"""

from __future__ import annotations

from weatherquant.market.auth import KalshiSigner, load_key, sign
from weatherquant.market.book import OrderBook, SeqGap, apply
from weatherquant.market.client import fetch_snapshot, run_feed
from weatherquant.market.clv import (
    CLV_WINDOW_MINUTES,
    clv_cents,
    closing_window_snapshots,
    vol_weighted_mid,
)
from weatherquant.market.persist import latest_snapshots, persist_fill, persist_snapshot
from weatherquant.market.reflect import no_ask_levels, yes_ask_levels

__all__ = [
    "KalshiSigner",
    "load_key",
    "sign",
    "yes_ask_levels",
    "no_ask_levels",
    "OrderBook",
    "SeqGap",
    "apply",
    "run_feed",
    "fetch_snapshot",
    "persist_snapshot",
    "persist_fill",
    "latest_snapshots",
    "CLV_WINDOW_MINUTES",
    "closing_window_snapshots",
    "vol_weighted_mid",
    "clv_cents",
]
