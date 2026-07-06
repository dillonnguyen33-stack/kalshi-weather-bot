#!/usr/bin/env python
"""Post a compact Discord summary of a GATE1-VERDICT.json to DISCORD_WEBHOOK_URL.

Usage: uv run python scripts/post_verdict_discord.py reports/verify_2026-07-04/GATE1-VERDICT.json
Silently no-ops (exit 0) when DISCORD_WEBHOOK_URL is unset, so a missing webhook never fails
the daily_verify.sh scoring run it's tacked onto.
"""

import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    verdict = json.loads(open(sys.argv[1]).read())
    webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    if not webhook:
        print("post_verdict_discord: DISCORD_WEBHOOK_URL unset, skipping")
        return

    window = verdict["test_window"]
    not_scored = verdict.get("not_scored", {})
    lines = [
        f"{'PASS' if verdict['passed'] else 'FAIL'} — Gate-1 verdict for {window[0]}",
        "",
    ]
    for metric, (lo, hi) in verdict["cis"].items():
        tag = " (not scored)" if not_scored.get(metric) else ""
        lines.append(f"{metric:>6}: [{lo:+.4f}, {hi:+.4f}]{tag}")
    content = "```\n" + "\n".join(lines) + "\n```"

    resp = httpx.post(webhook, json={"content": content}, timeout=10)
    resp.raise_for_status()


if __name__ == "__main__":
    main()
