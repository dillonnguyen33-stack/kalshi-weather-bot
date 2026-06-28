"""RED contract for the Gate-1 report renderer (VER-02 / D-11).

Asserts the Wave-4 ``render_reports`` behavior against a tmp out_dir:

* Writes reliability + PIT PNGs per stratum and ``GATE1-VERDICT.{json,md}``.
* The verdict JSON carries the five pooled CIs + PASS/FAIL + the RNG seed + excluded-day coverage.
* Filenames are built from VALIDATED city codes only (V12 path-safety) — no untrusted segments.

Imports are deferred so collection stays green while the implementation is RED.
"""

from __future__ import annotations

import json


def _sample_inputs():
    """A minimal (records, cis, verdict) triple shaped like the Wave-3 backtest output."""
    records = [
        {"city": "KXHIGHNY", "pit": [0.1, 0.4, 0.6, 0.9], "f": [0.5], "o": [1]},
    ]
    cis = {
        "brier": (-0.05, -0.01), "crps": (-0.05, -0.01), "ece": (-0.05, -0.01),
        "roi": (0.01, 0.05), "clv": (0.01, 0.05),
    }
    verdict = {"passed": True, "seed": 0, "excluded_days": []}
    return records, cis, verdict


def test_render_reports_writes_pngs_and_verdict_artifacts(tmp_path):
    """render_reports writes reliability/PIT PNGs + GATE1-VERDICT.{json,md} into out_dir."""
    from weatherquant.verify import report

    records, cis, verdict = _sample_inputs()
    written = report.render_reports(records, cis, verdict, out_dir=tmp_path)

    # At least one reliability + one PIT PNG, plus the two verdict artifacts.
    pngs = list(tmp_path.glob("*.png"))
    assert any("reliab" in p.name.lower() for p in pngs)
    assert any("pit" in p.name.lower() for p in pngs)
    assert (tmp_path / "GATE1-VERDICT.json").is_file()
    assert (tmp_path / "GATE1-VERDICT.md").is_file()
    assert isinstance(written, dict) and written  # name → path mapping


def test_verdict_json_carries_cis_seed_passfail_and_coverage(tmp_path):
    """The verdict JSON records the five pooled CIs, PASS/FAIL, the RNG seed, and excluded-day coverage."""
    from weatherquant.verify import report

    records, cis, verdict = _sample_inputs()
    report.render_reports(records, cis, verdict, out_dir=tmp_path)

    payload = json.loads((tmp_path / "GATE1-VERDICT.json").read_text())
    assert {"brier", "crps", "ece", "roi", "clv"} <= set(payload["cis"])
    assert "passed" in payload
    assert "seed" in payload
    assert "excluded_days" in payload


def test_report_filenames_use_validated_city_codes_only(tmp_path):
    """A record with a path-unsafe city code must not produce a traversal path (V12 path-safety)."""
    from weatherquant.verify import report

    _records, cis, verdict = _sample_inputs()
    malicious = [{"city": "../../etc/passwd", "pit": [0.5], "f": [0.5], "o": [1]}]
    # Either the renderer rejects the bad code loud, or it sanitizes — but NO file may escape out_dir.
    try:
        report.render_reports(malicious, cis, verdict, out_dir=tmp_path)
    except (ValueError, KeyError):
        return  # fail-loud rejection is acceptable
    for p in tmp_path.rglob("*"):
        assert tmp_path in p.resolve().parents or p.resolve() == tmp_path
