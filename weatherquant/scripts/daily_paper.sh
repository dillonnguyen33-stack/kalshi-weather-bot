#!/usr/bin/env bash
# Daily Gate-1 paper-trading launcher (step 4 of the Gate-1 plan).
#
# Refreshes today's forecasts/obs/AFD (idempotent) then launches one `paper --watch` loop per
# target market so the fills ledger accumulates a real forward track record. Cron this once per
# trading day; the verify step later scores the window the fills cover.
#
#   Usage:  scripts/daily_paper.sh [targets-file]      (default: scripts/paper_targets.txt)
#   Cron:   0 13 * * *  cd /path/to/weatherquant && scripts/daily_paper.sh >> reports/cron.log 2>&1
#
# targets-file: one "CITY TICKER" per line; '#' and blank lines ignored. A literal "{kdate}" in
# the ticker is replaced with today's Kalshi date code (YYMMMDD upper, e.g. 26JUN30). The bucket
# (B<mid>/T<thresh>) still changes with the forecast, so you pick today's bucket per line.
#
# Watch loops run concurrently (each blocks to settlement) so multiple cities trade the same day;
# per-target logs land in reports/. If `weatherquant live` is already running as a daemon, the
# ingest call below is harmless (idempotent) — the script stays self-sufficient without it.
set -euo pipefail

cd "$(dirname "$0")/.."                       # repo root = weatherquant/
targets="${1:-scripts/paper_targets.txt}"
today="$(date +%F)"                           # LST settlement date, YYYY-MM-DD (the --date arg)
kdate="$(date +%y%b%d | tr '[:lower:]' '[:upper:]')"   # Kalshi ticker date code, e.g. 26JUN30
mkdir -p reports
stamp() { date -u +%FT%TZ; }

if [[ ! -f "$targets" ]]; then
  echo "[$(stamp)] no targets file at $targets — nothing to trade" >&2
  exit 1
fi

echo "[$(stamp)] daily paper run for ${today} (targets=$targets)"

# 1. Refresh today's inputs so pricing sees the latest cycle (idempotent re-run is a no-op).
uv run weatherquant ingest --all-models --all-cities --date "$today"

# 2. One watch loop per target, concurrent (each runs to settlement / its --max-duration cap).
pids=()
while read -r city ticker _rest; do
  [[ -z "${city// }" || "${city:0:1}" == "#" ]] && continue
  ticker="${ticker//\{kdate\}/$kdate}"
  tlog="reports/paper_${today}_${city}_${ticker}.log"
  echo "[$(stamp)] launch paper --watch city=$city ticker=$ticker → $tlog"
  uv run weatherquant paper --city "$city" --date "$today" --ticker "$ticker" --watch \
    >"$tlog" 2>&1 &
  pids+=("$!")
done < "$targets"

# 3. Wait for every watch loop; report any that exited non-zero (don't abort the others).
rc=0
for pid in "${pids[@]:-}"; do
  [[ -z "$pid" ]] && continue
  if ! wait "$pid"; then
    echo "[$(stamp)] WARN: a paper watch loop (pid $pid) exited non-zero — see reports/paper_${today}_*.log" >&2
    rc=1
  fi
done

echo "[$(stamp)] daily paper run complete (rc=$rc)"
exit "$rc"
