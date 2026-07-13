#!/usr/bin/env bash
# Daily scoring launcher — the automated analog of manually running `weatherquant verify` each
# morning. Scores YESTERDAY's trading day (all cities/models the daily_paper.sh watch loops
# settled overnight) into its OWN dated --out-dir.
#
# IMPORTANT: `weatherquant verify` (no --monitor) is the Gate-1 MILESTONE proof — it freezes a
# pre-registration on first run and hard-rejects any later run whose --start/--end differ
# (anti-p-hacking guard, by design). Reusing reports/ across days would collide with that guard.
# A fresh dated --out-dir sidesteps it cleanly: each day gets its own pre-registration + verdict,
# so this is an operational daily scorecard, NOT a replacement for the one-time frozen-window
# Gate-1 proof you'll still run separately over the full accumulated ledger before going live.
#
#   Usage:  scripts/daily_verify.sh [YYYY-MM-DD]      (default: yesterday, i.e. the day daily_paper.sh
#                                                       just finished settling overnight)
#   Cron:   0 6 * * *  cd /path/to/weatherquant && scripts/daily_verify.sh >> reports/cron.log 2>&1
set -euo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

cd "$(dirname "$0")/.."

start="${1:-$(date -v-1d +%F)}"    # the settled trading day (LST date daily_paper.sh traded)
end="$(date -j -f %F -v+1d "$start" +%F)"   # half-open: start's very next day
lead="${LEAD:-24}"
out_dir="reports/verify_${start}"

stamp() { date -u +%FT%TZ; }
echo "[$(stamp)] daily verify: scoring ${start} (window [$start, $end)) → $out_dir"

uv run weatherquant verify --all-models --all-cities --lead "$lead" \
  --start "$start" --end "$end" --out-dir "$out_dir"

echo "[$(stamp)] daily verify complete — see ${out_dir}/GATE1-VERDICT.md"

# Discord summary — non-fatal (a webhook hiccup shouldn't fail an otherwise-scored day).
uv run python scripts/post_verdict_discord.py "${out_dir}/GATE1-VERDICT.json" \
  || echo "[$(stamp)] WARN: Discord post failed" >&2
