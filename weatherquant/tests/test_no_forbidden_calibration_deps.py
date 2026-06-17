"""Guard test: the calibration core must not import scipy/sklearn (PROJECT.md constraint).

The pure-NumPy boundary IS Phase 3: CRPS, its analytic gradient, the optimizer, pooling,
and the baseline are all hand-rolled NumPy with the normal CDF built from stdlib
``math.erf`` — never ``scipy.stats.norm`` (the "natural" reach) or any sklearn estimator.
``scipy`` / ``sklearn`` are a hard CLAUDE.md / PROJECT.md constraint (deploy stays
lightweight) and a subtle, easy-to-violate one, so this AST guard fences them out.

This mirrors ``tests/test_no_forbidden_runtime_deps.py`` exactly, changing only two things:
the scan root is scoped to the ``weatherquant.calibrate`` subpackage path (not the whole
runtime package) and ``FORBIDDEN`` is ``{"scipy", "sklearn"}``. It is a real source scan,
not a tautology: a stray ``import scipy`` under ``calibrate/`` would resolve at runtime
(scipy is present transitively in the lockfile) and silently violate the rule.
"""

from __future__ import annotations

import ast
import pathlib

import weatherquant.calibrate

FORBIDDEN = {"scipy", "sklearn"}

_CALIBRATE_ROOT = pathlib.Path(weatherquant.calibrate.__file__).resolve().parent


def _calibrate_modules() -> list[pathlib.Path]:
    return sorted(_CALIBRATE_ROOT.rglob("*.py"))


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


def test_calibrate_package_has_no_forbidden_imports():
    offenders: dict[str, set[str]] = {}
    for path in _calibrate_modules():
        imported = _imported_top_level_modules(path.read_text(encoding="utf-8"))
        leaked = imported & FORBIDDEN
        if leaked:
            offenders[str(path.relative_to(_CALIBRATE_ROOT))] = leaked
    assert not offenders, (
        f"forbidden libraries imported in the calibration core: {offenders} "
        f"(PROJECT.md / CLAUDE.md — pure NumPy + math.erf, never scipy/sklearn)."
    )
