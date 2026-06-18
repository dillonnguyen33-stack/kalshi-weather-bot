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
"""

from __future__ import annotations

from weatherquant.market.auth import KalshiSigner, load_key, sign
from weatherquant.market.book import OrderBook, SeqGap, apply
from weatherquant.market.client import fetch_snapshot, run_feed
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
]
