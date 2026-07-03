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

# cron/launchd run with a minimal PATH — prepend the dirs that hold uv (Homebrew) and docker so the
# scheduled invocation resolves them exactly like an interactive shell does.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

cd "$(dirname "$0")/.."                       # repo root = weatherquant/
targets="${1:-scripts/paper_targets.txt}"
today="$(date +%F)"                           # LST settlement date, YYYY-MM-DD (the --date arg)
kdate="$(date +%y%b%d | tr '[:lower:]' '[:upper:]')"   # Kalshi ticker date code, e.g. 26JUN30
lead="${LEAD:-24}"                            # trade horizon — MUST match the calibrated lead (24h-ahead high)
stagger="${STAGGER:-3}"                        # seconds between WS handshakes — Kalshi 429s a concurrent burst
auto_pick="${AUTO_PICK:-1}"                    # 1 = regenerate targets from today's forecast before launch
# paper --watch's own default cap is 4h, which truncates the loop mid-afternoon — before the
# daily-high peak and the pre-settlement closing window (no CLV). Pass a cap long enough that the
# settlement-window end is the binding bound instead (loops run to LST midnight, ~16-19h out). A
# machine that sleeps overnight can shorten this via WATCH_MAX_DURATION at the cost of CLV coverage.
watch_max="${WATCH_MAX_DURATION:-72000}"       # 20h — lets settlement bind for every city
mkdir -p reports
stamp() { date -u +%FT%TZ; }

echo "[$(stamp)] daily paper run for ${today} (targets=$targets)"

# 1. Refresh today's inputs at the TRADE lead so pricing/calibration align (idempotent re-run is a no-op).
uv run weatherquant ingest --all-models --all-cities --date "$today" --lead "$lead"

# 1b. Regenerate targets from today's fresh forecast (nearest open buckets per city). Only when the
# default file is in use — an explicit targets-file arg or AUTO_PICK=0 keeps a hand-picked list.
if [[ -z "${1:-}" && "$auto_pick" != "0" ]]; then
  echo "[$(stamp)] regenerating targets via pick_targets → $targets"
  uv run python scripts/pick_targets.py --date "$today" --lead "$lead" --out "$targets"
fi

if [[ ! -f "$targets" ]]; then
  echo "[$(stamp)] no targets file at $targets — nothing to trade" >&2
  exit 1
fi

# 2. One watch loop per target, concurrent (each runs to settlement / its --max-duration cap).
pids=()
while read -r city ticker _rest; do
  [[ -z "${city// }" || "${city:0:1}" == "#" ]] && continue
  ticker="${ticker//\{kdate\}/$kdate}"
  tlog="reports/paper_${today}_${city}_${ticker}.log"
  echo "[$(stamp)] launch paper --watch city=$city ticker=$ticker → $tlog"
  uv run weatherquant paper --city "$city" --date "$today" --ticker "$ticker" --lead "$lead" --watch \
    --max-duration "$watch_max" >"$tlog" 2>&1 &
  pids+=("$!")
  sleep "$stagger"                             # space out handshakes so Kalshi doesn't 429 the burst
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
