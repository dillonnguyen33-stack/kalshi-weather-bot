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
"""

from __future__ import annotations

from weatherquant.market.auth import KalshiSigner, load_key, sign

__all__ = [
    "KalshiSigner",
    "load_key",
    "sign",
]
