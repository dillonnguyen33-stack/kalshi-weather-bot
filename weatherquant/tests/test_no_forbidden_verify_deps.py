"""Guard test: the verify METRIC CORE must not import scipy/sklearn (PROJECT.md constraint).

The pure-NumPy boundary extends to the Phase-6 Gate-1 proof core: the Brier/Murphy/ECE/PIT/CRPS
metrics, the paired day-block bootstrap, and the conjunctive gate logic are all hand-rolled NumPy +
stdlib, reusing the erf-based normal CDF / closed-form CRPS from ``weatherquant.calibrate.crps`` —
never ``scipy.stats`` (the "natural" reach for a CDF or a bootstrap CI) or any sklearn calibration
estimator. ``scipy`` / ``sklearn`` are a hard CLAUDE.md / PROJECT.md constraint (deploy stays
lightweight) and a subtle, easy-to-violate one, so this AST guard fences them out of the verify
metric core (D-07).

This is a clone of ``tests/test_no_forbidden_price_deps.py``, changing the scan root and narrowing
the scanned-module ALLOWLIST. Unlike the calibrate/ + price/ guards (which scan the whole subtree),
this one scans ONLY the pure numeric core — ``metrics.py``, ``bootstrap.py``, ``gate1.py``. It
deliberately EXCLUDES ``report.py`` (the SOLE legitimate matplotlib importer, D-11 reporting edge)
and the adapter/orchestration edge (``backtest.py``, ``drift.py``, ``v3_reference.py``), which may
read the ledger / shared price geometry. It is a real source scan, not a tautology: a stray
``import scipy`` in the core would resolve at runtime (scipy is present transitively) and silently
violate the rule.
"""

from __future__ import annotations

import ast
import pathlib

import weatherquant.verify

FORBIDDEN = {"scipy", "sklearn"}

_VERIFY_ROOT = pathlib.Path(weatherquant.verify.__file__).resolve().parent

# The pure numeric core ONLY (D-07). report.py is EXCLUDED (the sole matplotlib importer, D-11);
# backtest/drift/v3_reference are the adapter/orchestration edge and are excluded too. Keeping this
# an explicit allowlist (not an rglob over the whole subtree) is the whole point of the guard.
_CORE_MODULES = ("metrics.py", "bootstrap.py", "gate1.py")


def _core_module_paths() -> list[pathlib.Path]:
    return [_VERIFY_ROOT / name for name in _CORE_MODULES]


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


def test_verify_core_modules_exist():
    """The allowlisted core modules must all exist — a rename must not silently shrink the scan."""
    missing = [p.name for p in _core_module_paths() if not p.is_file()]
    assert not missing, f"verify metric-core modules missing from the scan allowlist: {missing}"


def test_report_is_excluded_from_the_core_scan():
    """report.py is deliberately OUTSIDE the scanned core (the SOLE matplotlib importer, D-11)."""
    assert "report.py" not in _CORE_MODULES
    assert (_VERIFY_ROOT / "report.py").is_file()


def test_verify_core_has_no_forbidden_imports():
    offenders: dict[str, set[str]] = {}
    for path in _core_module_paths():
        imported = _imported_top_level_modules(path.read_text(encoding="utf-8"))
        leaked = imported & FORBIDDEN
        if leaked:
            offenders[path.name] = leaked
    assert not offenders, (
        f"forbidden libraries imported in the verify metric core: {offenders} "
        f"(PROJECT.md / CLAUDE.md — pure NumPy + math.erf, never scipy/sklearn)."
    )
