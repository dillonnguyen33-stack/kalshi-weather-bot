"""Gate-1 report rendering (VER-02 / D-11): reliability + PIT PNGs and the GATE1-VERDICT artifacts.

D-11 (verify subtree-local; matplotlib isolation): this is the SOLE module in the verify subtree
that may import matplotlib — and the import is kept LAZY inside :func:`render_reports` (with the Agg
backend forced BEFORE ``pyplot`` so it renders headless in CI) so the pure-NumPy metric core
(``metrics``/``bootstrap``/``gate1``) never pays for it and the no-forbidden AST guard fences
scipy/sklearn out of the core while explicitly EXCLUDING this reporting edge.

``render_reports`` writes per-stratum (per city, NOT just pooled — RESEARCH §Pitfall 4: pooling
hides per-city errors) reliability + PIT-histogram PNGs and the ``GATE1-VERDICT.{json,md}`` (the
five pooled CIs + PASS/FAIL up top, the per-stratum secondaries below, the RNG seed, the test
window/lead, and excluded-day coverage — fragility made visible, RESEARCH §State of the Art) plus
the frozen pre-registration into a project-relative, gitignored ``reports/`` dir. The MD mirror
carries the D-03 v3-exclusion note (RESEARCH §Pitfall 6). Output filenames are built from VALIDATED
city codes only (V12 path-safety) — no untrusted path segment is interpolated into a path.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["render_reports"]

# The five pre-registered Gate-1 metric keys the verdict JSON must carry (mirrors gate1.GATE1_METRICS).
_GATE1_METRICS = ("brier", "crps", "ece", "roi", "clv")

# A city code is path-safe iff it is alphanumeric (Kalshi codes like KXHIGHNY) — anything else
# (slashes, dots, traversal segments) is rejected loud so no filename can escape ``out_dir`` (V12).
_SAFE_CITY = re.compile(r"\A[A-Za-z0-9_-]+\Z")

# Equal-width bin count for the reliability-diagram x-axis (the diagram grid — distinct from the
# equal-count ECE scalar; RESEARCH §Pitfall 4).
_RELIABILITY_BINS = 10
# PIT-histogram bin count (a flat histogram = a calibrated forecast).
_PIT_BINS = 10


def _safe_city(city: str) -> str:
    """Return ``city`` only if it is a path-safe code; otherwise fail LOUD (V12 path-safety).

    Filenames are built from validated city codes ONLY — a traversal segment (``../etc/passwd``)
    or any non-alphanumeric path character is rejected with ``ValueError`` rather than interpolated
    into a path that could escape ``out_dir``.
    """
    if not isinstance(city, str) or not _SAFE_CITY.match(city):
        raise ValueError(
            f"unsafe city code {city!r}: report filenames are built from validated "
            f"alphanumeric city codes only (V12 path-safety) — refusing to write."
        )
    return city


def render_reports(
    records, cis, verdict, *, out_dir, seed: int | None = None, coverage=None
) -> dict[str, str]:
    """Render the Gate-1 reliability/PIT PNGs + GATE1-VERDICT.{json,md} into ``out_dir`` (VER-02).

    matplotlib is imported LAZILY inside this body (D-11 isolation) — never at module top — with the
    Agg backend forced BEFORE ``pyplot`` so the PNGs render headless in CI. Per stratum (per city)
    this writes ``reliability_<city>.png`` (equal-width predicted-prob bins on ``[0, 1]`` vs observed
    YES frequency, the 45° reference line + a bin-count histogram beneath) and ``pit_<city>.png``
    (the PIT histogram — flat = calibrated). It then writes ``GATE1-VERDICT.json`` (the five pooled
    CIs + PASS/FAIL + per-stratum secondaries + RNG seed + test window/lead + excluded-day coverage)
    and its human-readable ``GATE1-VERDICT.md`` mirror (pooled PASS/FAIL up top, secondaries below,
    coverage logged, the D-03 v3-exclusion note), plus the frozen ``gate1_preregistration.json`` if a
    pre-registration spec rides on the verdict. Returns a mapping of artifact name → written path.

    Filenames are built from validated city codes ONLY (V12 path-safety): an unsafe code fails loud
    rather than being interpolated into a path that could escape ``out_dir``.
    """
    # D-11: matplotlib lives here and ONLY here, lazily — force the headless Agg backend BEFORE
    # importing pyplot so the renderer never needs a display (CI-safe).
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    # Resolve the RNG seed (explicit arg wins; else the verdict's, else 0) — recorded so the
    # bootstrap is reproducible from the verdict alone (anti-p-hacking provenance).
    if seed is None:
        seed = int(verdict.get("seed", 0)) if isinstance(verdict, dict) else 0

    # Excluded-day coverage: prefer the explicit coverage arg, else the verdict's excluded_days.
    if coverage is None:
        coverage = verdict.get("excluded_days", []) if isinstance(verdict, dict) else []

    # --- Per-stratum PNGs (per city, NOT just pooled — RESEARCH §Pitfall 4) -------------------
    pooled_f: list[float] = []
    pooled_o: list[float] = []
    pooled_pit: list[float] = []
    for rec in records:
        city = _safe_city(rec["city"])  # fail loud on an unsafe code BEFORE any path is built
        f = np.asarray(rec.get("f", []), dtype=float)
        o = np.asarray(rec.get("o", []), dtype=float)
        pit = np.asarray(rec.get("pit", []), dtype=float)
        pooled_f.extend(f.tolist())
        pooled_o.extend(o.tolist())
        pooled_pit.extend(pit.tolist())

        rel_path = out / f"reliability_{city}.png"
        _render_reliability(plt, np, f, o, city, rel_path)
        written[f"reliability_{city}"] = str(rel_path)

        pit_path = out / f"pit_{city}.png"
        _render_pit(plt, np, pit, city, pit_path)
        written[f"pit_{city}"] = str(pit_path)

    # A pooled diagram too, so the verdict shows both the pooled headline and the per-city detail.
    if pooled_f:
        pooled_rel = out / "reliability_pooled.png"
        _render_reliability(plt, np, np.asarray(pooled_f), np.asarray(pooled_o), "pooled", pooled_rel)
        written["reliability_pooled"] = str(pooled_rel)
    if pooled_pit:
        pooled_pit_path = out / "pit_pooled.png"
        _render_pit(plt, np, np.asarray(pooled_pit), "pooled", pooled_pit_path)
        written["pit_pooled"] = str(pooled_pit_path)

    # --- Verdict artifacts (fragility visible: pooled PASS/FAIL up top, secondaries below) -----
    passed = bool(verdict.get("passed")) if isinstance(verdict, dict) else bool(verdict)
    cis_payload = {k: list(cis[k]) for k in cis}
    payload: dict[str, Any] = {
        "passed": passed,
        "cis": cis_payload,
        "seed": int(seed),
        "excluded_days": list(coverage),
    }
    # Carry through the optional provenance the caller front-loads onto the verdict (window/lead/
    # secondaries) so the JSON is self-describing — without inventing keys the caller did not pass.
    if isinstance(verdict, dict):
        for key in ("test_window", "primary_lead", "secondaries", "preregistration"):
            if key in verdict:
                payload[key] = verdict[key]

    json_path = out / "GATE1-VERDICT.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    written["verdict_json"] = str(json_path)

    md_path = out / "GATE1-VERDICT.md"
    md_path.write_text(_render_verdict_md(payload))
    written["verdict_md"] = str(md_path)

    # The frozen pre-registration mirror, if the caller threaded a spec onto the verdict (D-08/D-13).
    prereg = verdict.get("preregistration") if isinstance(verdict, dict) else None
    if prereg:
        prereg_path = out / "gate1_preregistration.json"
        prereg_path.write_text(json.dumps(prereg, indent=2, sort_keys=True, default=str))
        written["preregistration"] = str(prereg_path)

    logger.info(
        "render_reports wrote %d artifact(s) to %s (passed=%s, excluded=%d)",
        len(written),
        out,
        passed,
        len(payload["excluded_days"]),
    )
    return written


def _render_reliability(plt, np, f, o, city: str, path: Path) -> None:
    """Reliability diagram (equal-width predicted-prob bins vs observed YES freq) + bin-count hist.

    Equal-WIDTH bins on ``[0, 1]`` (the diagram x-axis — distinct from the equal-count ECE scalar,
    RESEARCH §Pitfall 4): per bin, the mean predicted probability (x) vs the observed YES frequency
    (y), against the 45° perfect-calibration reference, with a bin-population histogram beneath so a
    sparse bin is visibly down-weighted.
    """
    edges = np.linspace(0.0, 1.0, _RELIABILITY_BINS + 1)
    centers = []
    observed = []
    counts = []
    if f.size:
        idx = np.clip(np.digitize(f, edges[1:-1]), 0, _RELIABILITY_BINS - 1)
        for k in range(_RELIABILITY_BINS):
            mask = idx == k
            n_k = int(mask.sum())
            counts.append(n_k)
            if n_k:
                centers.append(float(f[mask].mean()))
                observed.append(float(o[mask].mean()))
    else:
        counts = [0] * _RELIABILITY_BINS

    fig, (ax_rel, ax_hist) = plt.subplots(
        2, 1, figsize=(5, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    ax_rel.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect")
    if centers:
        ax_rel.plot(centers, observed, marker="o", label="observed")
    ax_rel.set_ylabel("observed YES frequency")
    ax_rel.set_title(f"Reliability — {city}")
    ax_rel.set_xlim(0, 1)
    ax_rel.set_ylim(0, 1)
    ax_rel.legend(loc="best")

    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    ax_hist.bar(bin_centers, counts, width=1.0 / _RELIABILITY_BINS, align="center", color="steelblue")
    ax_hist.set_ylabel("count")
    ax_hist.set_xlabel("predicted probability")

    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _render_pit(plt, np, pit, city: str, path: Path) -> None:
    """PIT histogram (flat = calibrated; U-shaped = overdispersed, dome = overconfident)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    if pit.size:
        ax.hist(pit, bins=_PIT_BINS, range=(0.0, 1.0), color="steelblue", edgecolor="white")
    ax.axhline(
        (pit.size / _PIT_BINS) if pit.size else 0.0,
        linestyle="--",
        color="gray",
        label="uniform",
    )
    ax.set_title(f"PIT histogram — {city}")
    ax.set_xlabel("PIT value")
    ax.set_ylabel("count")
    ax.set_xlim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _render_verdict_md(payload: dict[str, Any]) -> str:
    """Human-readable verdict mirror: pooled PASS/FAIL up top, secondaries below, coverage logged."""
    verdict_word = "PASS" if payload["passed"] else "FAIL"
    lines = [
        "# Gate-1 Verdict",
        "",
        f"**Overall: {verdict_word}**",
        "",
        "## Pooled metric CIs (95% paired day-block bootstrap, weatherquant − v3)",
        "",
        "| Metric | CI low | CI high |",
        "| ------ | ------ | ------- |",
    ]
    for metric in _GATE1_METRICS:
        if metric in payload["cis"]:
            lo, hi = payload["cis"][metric]
            lines.append(f"| {metric} | {lo} | {hi} |")
    lines += [
        "",
        f"- RNG seed: `{payload['seed']}`",
    ]
    if payload.get("primary_lead") is not None:
        lines.append(f"- Primary lead: `{payload['primary_lead']}`")
    if payload.get("test_window") is not None:
        lines.append(f"- Test window: `{payload['test_window']}`")

    secondaries = payload.get("secondaries")
    if secondaries:
        lines += ["", "## Per-stratum secondaries (Holm-adjusted)", "", "```json",
                   json.dumps(secondaries, indent=2, sort_keys=True, default=str), "```"]

    excluded = payload.get("excluded_days", [])
    lines += [
        "",
        f"## Excluded-day coverage ({len(excluded)} day(s))",
        "",
    ]
    if excluded:
        lines += ["```json", json.dumps(excluded, indent=2, sort_keys=True, default=str), "```"]
    else:
        lines.append("_No days excluded._")

    lines += [
        "",
        "## Note (D-03 v3-exclusion)",
        "",
        "The v3 reference arm reproduces ONLY the legacy ENSEMBLE branch (no ASOS override, no "
        "is-next-day branch, no threshold branch) and prices on a point-in-time bias measured from "
        "`available_at < cutoff` outcomes only — the comparison is leak-safe by construction "
        "(RESEARCH §Pitfall 6).",
        "",
    ]
    return "\n".join(lines)
