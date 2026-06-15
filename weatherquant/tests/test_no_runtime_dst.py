"""Guard test: runtime settlement modules must NOT import zoneinfo/timezonefinder.

TIME-01 / Pitfall 1 — the v3 settlement bug was DST-aware tz on the runtime path. This
test AST-scans ``weatherquant/time.py`` and ``weatherquant/registry.py`` and asserts
neither imports ``zoneinfo`` nor ``timezonefinder`` (those belong only in tests/tooling,
e.g. ``test_registry.py``'s D-02 cross-check).

RED until plan 01-02: the modules do not exist yet, so resolving their source path
raises ``ModuleNotFoundError`` — the correct RED signal. Once 01-02 lands, the AST scan
runs and enforces the no-DST contract.
"""

from __future__ import annotations

import ast
import importlib.util

import pytest

RUNTIME_MODULES = ["weatherquant.time", "weatherquant.registry"]
FORBIDDEN_RUNTIME_IMPORTS = {"zoneinfo", "timezonefinder"}


def _module_source_path(module_name: str) -> str:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        # RED reason at 01-01: module not written yet.
        raise ModuleNotFoundError(
            f"{module_name} not found — implement it in plan 01-02 "
            f"(this guard then enforces the no-DST contract)."
        )
    return spec.origin


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


@pytest.mark.parametrize("module_name", RUNTIME_MODULES)
def test_runtime_module_has_no_dst_imports(module_name):
    path = _module_source_path(module_name)  # raises ModuleNotFoundError if absent (RED)
    with open(path, "r", encoding="utf-8") as fh:
        imported = _imported_top_level_modules(fh.read())
    leaked = imported & FORBIDDEN_RUNTIME_IMPORTS
    assert not leaked, (
        f"{module_name} imports forbidden DST tooling on the runtime path: {leaked} "
        f"(TIME-01 / Pitfall 1 — confine zoneinfo/timezonefinder to tests)."
    )
