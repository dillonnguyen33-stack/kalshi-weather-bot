"""RED stub — ING-07 / D-13: AFD pre-filter skips routine text (no paid API call).

Turned GREEN by 02-04's ``weatherquant.ingest.afd``. The keyword pre-filter
(``afd_should_classify``) runs BEFORE any Anthropic call to stay in budget: routine AFDs
(``near normal``) are skipped; signal AFDs (``model disagreement``) are classified via the
SDK forced-``tool_choice`` pattern. Uses the vendored AFD text fixtures. RED at import
until 02-04.
"""

from __future__ import annotations

import pathlib

_FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


def test_prefilter_skips_routine_classifies_signal():
    # RED: weatherquant.ingest.afd lands in 02-04 (ImportError until then).
    from weatherquant.ingest.afd import afd_should_classify

    routine = (_FIXTURES / "afd_sample_routine.txt").read_text()
    signal = (_FIXTURES / "afd_sample_signal.txt").read_text()

    should_routine, _ = afd_should_classify(routine)
    should_signal, _ = afd_should_classify(signal)
    assert should_routine is False  # routine text → no paid call (D-13)
    assert should_signal is True  # signal text → classify
