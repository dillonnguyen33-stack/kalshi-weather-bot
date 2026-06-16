"""Guard test: the runtime package must not import legacy drivers (D-09 / Pitfall 4).

The stack mandates psycopg v3 and httpx; ``psycopg2`` and ``requests`` are forbidden on
the active path (psycopg2 is legacy/sync-only, requests clashes with the asyncio design).
``requests`` is present transitively in the lockfile via herbie-data, so a stray
``import requests`` would resolve at runtime and silently violate the rule. The psycopg3
scheme is enforced by the engine validator, but neither driver had an import-level guard.

This AST-scans every module under the ``weatherquant`` package (not tests/tooling) and
asserts none imports ``psycopg2`` or ``requests`` at top level. It mirrors
``test_no_runtime_dst.py``: a real source scan, not a tautology. ``psycopg`` (v3) is
fine — only the exact ``psycopg2`` top-level name is rejected.
"""

from __future__ import annotations

import ast
import pathlib

import weatherquant

FORBIDDEN_RUNTIME_IMPORTS = {"psycopg2", "requests"}

_PACKAGE_ROOT = pathlib.Path(weatherquant.__file__).resolve().parent


def _runtime_modules() -> list[pathlib.Path]:
    return sorted(_PACKAGE_ROOT.rglob("*.py"))


def _imported_top_level_modules(source: str) -> set[str]:
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module.split(".")[0])
    return names


def test_runtime_package_has_no_forbidden_driver_imports():
    offenders: dict[str, set[str]] = {}
    for path in _runtime_modules():
        imported = _imported_top_level_modules(path.read_text(encoding="utf-8"))
        leaked = imported & FORBIDDEN_RUNTIME_IMPORTS
        if leaked:
            offenders[str(path.relative_to(_PACKAGE_ROOT))] = leaked
    assert not offenders, (
        f"forbidden legacy drivers imported on the runtime path: {offenders} "
        f"(D-09 / Pitfall 4 — use psycopg v3 and httpx, never psycopg2/requests)."
    )
