"""Pure-NumPy EMOS/NGR calibration core: a 4-param Gaussian per ``(city, model, lead, month)``.

No scipy/sklearn anywhere under this package — a hard PROJECT.md/CLAUDE.md rule enforced by
the AST guard ``tests/test_no_forbidden_calibration_deps.py`` (see docs/DECISIONS.md).
"""

from __future__ import annotations
