"""Kalshi market edge for the paper-fill simulator (Phase 5).

The I/O edge that talks to Kalshi (live orderbook WS feed, REST snapshot, RSA-PSS signing)
plus the pure correctness seams the fill simulator prices against. Fenced OUT of the pure
``weatherquant.price`` money path by ``tests/test_no_market_leak_into_price.py``.

Public surface:

* ``auth`` — :class:`KalshiSigner`, :func:`load_key`, :func:`sign`: the ONE RSA-PSS signer
  shared by the WS handshake and the REST snapshot (PAP-01, D-01).
* ``reflect`` — :func:`yes_ask_levels`, :func:`no_ask_levels`: the ONE yes/no bid-only
  reflection seam (``ask = 100 - opposite bid``) every fill price/size routes through (PAP-02).
* ``book`` — :class:`OrderBook`, :class:`SeqGap`, :func:`apply`: the in-memory per-ticker
  book + ``seq`` integrity. A gap raises ``SeqGap`` (a ``CorrectnessError``) — never a silent
  carry-forward; it breaks the delta loop so the feed reconnects and re-anchors on a fresh WS
  snapshot (PAP-01, D-02).
* ``client`` — :func:`run_feed`, :func:`fetch_snapshot`: the signed WS connect + manual
  reconnect loop. Each (re)connection re-signs the handshake and re-subscribes; the fresh WS
  ``orderbook_snapshot`` (seq=1) is the ONLY integrity anchor — NO REST re-snapshot on
  reconnect (B1/D-02). :func:`fetch_snapshot` is a separate signed one-shot REST read for
  ``run_paper`` (REST carries no ``seq``, so it is not a seq anchor — MED-5). (PAP-01)
* ``persist`` — :func:`persist_snapshot`, :func:`persist_fill`, :func:`latest_snapshots`:
  the THIN snapshot/fill adapter over the audited append-only writer + ``queries.latest``
  (no Core insert here — D-13, T-05-15).
* ``clv`` — :func:`clv_cents`, :func:`vol_weighted_mid`, :func:`closing_window_snapshots`,
  :data:`CLV_WINDOW_MINUTES`: the PURE per-trade CLV against a volume-weighted closing mid on
  the LST ``settlement_window`` clock (PAP-04, D-09/D-10/D-12).
"""

from __future__ import annotations

from weatherquant.market.auth import KalshiSigner, load_key, sign
from weatherquant.market.book import OrderBook, SeqGap, apply
from weatherquant.market.client import fetch_snapshot, run_feed
from weatherquant.market.clv import (
    CLV_WINDOW_MINUTES,
    closing_window_snapshots,
    clv_cents,
    vol_weighted_mid,
)
from weatherquant.market.persist import latest_snapshots, persist_fill, persist_snapshot
from weatherquant.market.reflect import no_ask_levels, yes_ask_levels

__all__ = [
    "CLV_WINDOW_MINUTES",
    "KalshiSigner",
    "OrderBook",
    "SeqGap",
    "apply",
    "closing_window_snapshots",
    "clv_cents",
    "fetch_snapshot",
    "latest_snapshots",
    "load_key",
    "no_ask_levels",
    "persist_fill",
    "persist_snapshot",
    "run_feed",
    "sign",
    "vol_weighted_mid",
    "yes_ask_levels",
]
