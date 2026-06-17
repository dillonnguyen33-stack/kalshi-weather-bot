"""Guard test: the price/ money path must not import scipy/sklearn (PROJECT.md constraint).

The pure-NumPy boundary extends to Phase 4: the blend, bucket CDF differencing, exact fee,
EV, and fractional Kelly are all hand-rolled NumPy + stdlib ``math``, reusing the erf-based
normal CDF promoted public in ``weatherquant.calibrate.crps`` — never ``scipy.stats.norm``
(the "natural" reach) or any sklearn estimator. ``scipy`` / ``sklearn`` are a hard CLAUDE.md
/ PROJECT.md constraint (deploy stays lightweight) and a subtle, easy-to-violate one, so this
AST guard fences them out of ``weatherquant.price`` (D-14).

This is a verbatim clone of ``tests/test_no_forbidden_calibration_deps.py``, changing only the
scan root (``weatherquant.price`` instead of ``weatherquant.calibrate``). ``FORBIDDEN`` stays
``{"scipy", "sklearn"}`` and the AST-walk + offenders assertion are reused as-is. It is a real
source scan, not a tautology: a stray ``import scipy`` under ``price/`` would resolve at
runtime (scipy is present transitively in the lockfile) and silently violate the rule.
"""

from __future__ import annotations

import ast
import pathlib

import weatherquant.price

FORBIDDEN = {"scipy", "sklearn"}

_PRICE_ROOT = pathlib.Path(weatherquant.price.__file__).resolve().parent


def _price_modules() -> list[pathlib.Path]:
    return sorted(_PRICE_ROOT.rglob("*.py"))


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


def test_price_package_has_no_forbidden_imports():
    offenders: dict[str, set[str]] = {}
    for path in _price_modules():
        imported = _imported_top_level_modules(path.read_text(encoding="utf-8"))
        leaked = imported & FORBIDDEN
        if leaked:
            offenders[str(path.relative_to(_PRICE_ROOT))] = leaked
    assert not offenders, (
        f"forbidden libraries imported in the price money path: {offenders} "
        f"(PROJECT.md / CLAUDE.md — pure NumPy + math.erf, never scipy/sklearn)."
    )
