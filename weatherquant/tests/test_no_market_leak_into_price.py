"""Guard test: the pure price/ core must not import the market/ I/O edge (no-leak boundary).

Phase 5 adds the ``weatherquant.market`` package — the I/O edge (signed WS feed, RSA-PSS
signer, orderbook book) that legitimately imports ``websockets`` / ``cryptography``. That edge
must NOT leak into the pure-NumPy ``weatherquant.price`` money path: the price core stays a
pure function of (forecast, market-midpoint) with no I/O dependency, so it remains fully
offline-unit-testable and the dependency arrow points one way (the CLI / market layer feeds
the midpoint INTO price, never the reverse).

This is a verbatim clone of ``tests/test_no_forbidden_price_deps.py``, changing only the
``FORBIDDEN`` set to the I/O-edge leaf tokens ``{"market", "websockets", "cryptography"}`` and
keeping the ``weatherquant.price`` scan root. ``market`` matches the leaf token the AST walk
extracts from ``import weatherquant.market`` (split on ``.`` → ``weatherquant``) is too broad,
so we match ``from weatherquant.market import ...`` (module leaf) and a bare ``import market``
by inspecting BOTH the top-level token and the second segment of any ``weatherquant.market``
import. It is a real source scan, not a tautology: a stray ``from weatherquant.market import
...`` or ``import websockets`` under ``price/`` would resolve at runtime and silently violate
the rule. Passes NOW (price/ imports no market symbol) and keeps passing as price/ grows.

NOTE: ``market/reflect.py`` / ``fills.py`` / ``clv.py`` stay PURE but live UNDER ``market/`` —
this guard targets the ``price/`` scan root ONLY. ``market/`` is exempt from the no-scipy guard
and ``price/`` is exempt from importing ``market``; the two boundaries are independent.
"""

from __future__ import annotations

import ast
import pathlib

import weatherquant.price

FORBIDDEN = {"market", "websockets", "cryptography"}

_PRICE_ROOT = pathlib.Path(weatherquant.price.__file__).resolve().parent


def _price_modules() -> list[pathlib.Path]:
    return sorted(_PRICE_ROOT.rglob("*.py"))


def _imported_tokens(source: str) -> set[str]:
    """Return the import tokens to police: top-level module name AND, for a
    ``weatherquant.market`` import, the ``market`` leaf — so ``import weatherquant.market``
    and ``from weatherquant.market import ...`` are both caught without flagging every
    ``weatherquant.*`` import."""
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                names.add(parts[0])
                if parts[0] == "weatherquant" and len(parts) > 1:
                    names.add(parts[1])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                parts = node.module.split(".")
                names.add(parts[0])
                if parts[0] == "weatherquant" and len(parts) > 1:
                    names.add(parts[1])
    return names


def test_price_package_does_not_import_market_or_io_edge():
    offenders: dict[str, set[str]] = {}
    for path in _price_modules():
        imported = _imported_tokens(path.read_text(encoding="utf-8"))
        leaked = imported & FORBIDDEN
        if leaked:
            offenders[str(path.relative_to(_PRICE_ROOT))] = leaked
    assert not offenders, (
        f"I/O-edge imports leaked into the pure price money path: {offenders} "
        f"(the price core must not import weatherquant.market / websockets / cryptography — "
        f"the dependency arrow points market→price, never the reverse)."
    )
