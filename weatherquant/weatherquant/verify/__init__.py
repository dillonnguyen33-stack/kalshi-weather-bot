"""Gate-1 verification edge package (D-01): pure-NumPy metric core, matplotlib isolated to report.

D-01 (verify subtree-local; the verify/ D-id namespace STARTS here at D-01): the verify/ package
is the terminal Gate-1 proof edge. Its numeric core (``metrics``, ``bootstrap``, ``gate1``) is
pure NumPy + stdlib — no scipy/sklearn (fenced by ``tests/test_no_forbidden_verify_deps.py``,
mirroring the calibrate/ + price/ guards). matplotlib is allowed ONLY in ``report.py`` (the
reporting edge, D-11), imported lazily inside the render function so the metric core never pays
for it. The adapter/orchestration modules (``v3_reference``, ``backtest``, ``drift``) sit at the
edge and may read the ledger / shared price geometry; they import nothing forbidden.
"""

from __future__ import annotations
