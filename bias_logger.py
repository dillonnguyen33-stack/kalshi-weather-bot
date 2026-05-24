"""
bias_logger.py — Kalshi Weather Bot prediction tracker

Sits next to kalshi_bot_v3.2.py. Three jobs:
  1. log_prediction()  — called by the bot on every alert, writes to SQLite
  2. score_settlements() — run manually or on a cron, fetches Kalshi results
                           and marks each prediction as won/lost
  3. calibration_report() — prints a table showing where the model is sharp,
                             overconfident, or underconfident

Database: ./predictions.db (SQLite, auto-created on first run)

Usage:
  # Score yesterday's markets (run after 11pm ET when all markets settled):
  python3 bias_logger.py score

  # Print calibration report:
  python3 bias_logger.py report

  # Report for a specific city:
  python3 bias_logger.py report NYC

  # Report for last N days:
  python3 bias_logger.py report --days 30
"""

import sqlite3, requests, json, sys, os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

DB_PATH       = os.path.join(os.path.dirname(__file__), "predictions.db")
KALSHI_BASE   = "https://external-api.kalshi.com/trade-api/v2"
ET_TZ         = ZoneInfo("America/New_York")

# ── SCHEMA ────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at       TEXT NOT NULL,          -- ISO timestamp when bot fired alert
    market_date     TEXT NOT NULL,          -- YYYY-MM-DD settlement date
    city_code       TEXT NOT NULL,          -- e.g. NY, CHI, LAX
    city_name       TEXT NOT NULL,
    ticker          TEXT NOT NULL UNIQUE,   -- Kalshi ticker, unique per market
    threshold_f     REAL NOT NULL,          -- temperature threshold in °F
    threshold_kind  TEXT NOT NULL,          -- 'T' (above/below) or 'B' (bucket)

    -- Model outputs
    model_prob      REAL NOT NULL,          -- our P(YES) at time of alert
    yes_price       INTEGER NOT NULL,       -- Kalshi YES ask in cents at alert time
    no_price        INTEGER NOT NULL,       -- Kalshi NO ask in cents at alert time
    best_side       TEXT,                   -- 'YES' or 'NO'
    taker_ev        REAL,                   -- fee-adjusted EV% at alert time
    ensemble_mean   REAL,
    spread          REAL,
    confidence      TEXT,
    bias_applied    REAL,
    asos_obs_high   REAL,                   -- ASOS observed high at alert time (if available)
    ecmwf_high      REAL,
    nbm_high        REAL,
    hrrr_high       REAL,
    icon_high       REAL,

    -- Settlement (filled in by score_settlements)
    settled         INTEGER DEFAULT 0,      -- 0 = pending, 1 = scored
    actual_high_f   REAL,                   -- actual NWS settlement high
    yes_result      INTEGER,                -- 1 = YES won, 0 = NO won, NULL = pending
    model_correct   INTEGER,                -- 1 = our best_side was correct
    closing_yes_price INTEGER,              -- Kalshi closing price (for CLV)
    clv             REAL                    -- closing line value: closing_price - our_entry_price
);

CREATE INDEX IF NOT EXISTS idx_market_date ON predictions(market_date);
CREATE INDEX IF NOT EXISTS idx_city        ON predictions(city_code);
CREATE INDEX IF NOT EXISTS idx_settled     ON predictions(settled);
"""

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn

# ── LOG PREDICTION ────────────────────────────────────────────────────────────

def log_prediction(
    market_date: date,
    city_code: str,
    city_name: str,
    ticker: str,
    threshold_f: float,
    forecast: dict,
    model_prob: float,
    yes_price: int,
    no_price: int,
    asos_obs_high: float | None,
    best_side: str | None = None,
    taker_ev: float | None = None,
    threshold_kind: str = "T",
):
    """
    Called by the bot on every scan that produces an alert.
    Silently skips duplicate tickers (same market fired again on rescan).
    """
    conn = get_db()
    try:
        conn.execute("""
            INSERT OR IGNORE INTO predictions (
                logged_at, market_date, city_code, city_name, ticker,
                threshold_f, threshold_kind, model_prob, yes_price, no_price,
                best_side, taker_ev,
                ensemble_mean, spread, confidence, bias_applied, asos_obs_high,
                ecmwf_high, nbm_high, hrrr_high, icon_high
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.now(ET_TZ).isoformat(),
            market_date.isoformat(),
            city_code, city_name, ticker,
            threshold_f, threshold_kind,
            model_prob, yes_price, no_price,
            best_side, taker_ev,
            forecast.get("ensemble_mean"),
            forecast.get("spread"),
            forecast.get("confidence"),
            forecast.get("bias_applied"),
            asos_obs_high,
            forecast.get("ecmwf_high"),
            forecast.get("nbm_high"),
            forecast.get("hrrr_high"),
            forecast.get("icon_high"),
        ))
        conn.commit()
    except Exception as e:
        print(f"[bias_logger] log error: {e}")
    finally:
        conn.close()

# ── SETTLEMENT SCORING ────────────────────────────────────────────────────────

def fetch_kalshi_result(ticker: str) -> dict | None:
    """
    Fetches the settled result for a Kalshi market.
    Returns {"yes_result": 1|0, "closing_yes_price": int} or None.
    """
    try:
        r = requests.get(f"{KALSHI_BASE}/markets/{ticker}", timeout=10)
        r.raise_for_status()
        m = r.json().get("market", {})

        status = m.get("status", "")
        if status not in ("settled", "finalized"):
            return None  # not settled yet

        result = m.get("result", "")
        yes_result = 1 if result == "yes" else (0 if result == "no" else None)
        if yes_result is None:
            return None

        # Closing price = last trade price before settlement
        closing = m.get("last_price")
        closing_cents = round(float(closing) * 100) if closing else None

        return {"yes_result": yes_result, "closing_yes_price": closing_cents}
    except Exception as e:
        print(f"[bias_logger] fetch result {ticker}: {e}")
        return None

def score_settlements(target_date: date | None = None):
    """
    Fetches Kalshi settlement results for all unscored predictions.
    By default scores all pending markets from yesterday and earlier.
    Run this once daily after ~11pm ET.
    """
    if target_date is None:
        target_date = date.today() - timedelta(days=1)

    conn = get_db()
    rows = conn.execute("""
        SELECT id, ticker, best_side, yes_price, model_prob
        FROM predictions
        WHERE settled = 0 AND market_date <= ?
        ORDER BY market_date
    """, (target_date.isoformat(),)).fetchall()

    if not rows:
        print(f"[bias_logger] No unsettled predictions for {target_date} or earlier")
        conn.close()
        return

    print(f"[bias_logger] Scoring {len(rows)} predictions...")
    scored = 0

    for row in rows:
        result = fetch_kalshi_result(row["ticker"])
        if result is None:
            print(f"  ⏳ {row['ticker']}: not settled yet, skipping")
            continue

        yes_result    = result["yes_result"]
        closing_price = result["closing_yes_price"]

        # Did our best_side win?
        model_correct = None
        if row["best_side"] == "YES":
            model_correct = 1 if yes_result == 1 else 0
            entry_price = row["yes_price"]
        elif row["best_side"] == "NO":
            model_correct = 1 if yes_result == 0 else 0
            entry_price = 100 - row["yes_price"]  # NO price in cents
        else:
            entry_price = None

        # CLV: closing price vs our entry (positive = we got a better price)
        clv = None
        if closing_price is not None and entry_price is not None:
            if row["best_side"] == "YES":
                clv = round(closing_price - entry_price, 1)
            else:
                closing_no = 100 - closing_price
                clv = round(closing_no - entry_price, 1)

        conn.execute("""
            UPDATE predictions
            SET settled=1, yes_result=?, model_correct=?, closing_yes_price=?, clv=?
            WHERE id=?
        """, (yes_result, model_correct, closing_price, clv, row["id"]))

        status = "✓" if model_correct else "✗"
        clv_str = f"CLV {clv:+.0f}¢" if clv is not None else "no CLV"
        print(f"  {status} {row['ticker']}: {'YES' if yes_result else 'NO'} won | {clv_str}")
        scored += 1

    conn.commit()
    conn.close()
    print(f"[bias_logger] Scored {scored}/{len(rows)} predictions")

# ── CALIBRATION REPORT ────────────────────────────────────────────────────────

def calibration_report(city_filter: str | None = None, days: int = 90):
    """
    Prints a calibration table: how accurate the model is at each probability
    bucket, and per-city breakdown of accuracy + CLV.

    A well-calibrated model at 70% confidence should win ~70% of the time.
    Systematic deviation = bias to correct.
    """
    conn = get_db()
    cutoff = (date.today() - timedelta(days=days)).isoformat()

    where = "settled = 1 AND market_date >= ?"
    params: list = [cutoff]
    if city_filter:
        where += " AND city_code = ?"
        params.append(city_filter.upper())

    rows = conn.execute(f"""
        SELECT city_code, city_name, model_prob, best_side, taker_ev,
               model_correct, clv, ensemble_mean, spread, market_date, ticker
        FROM predictions WHERE {where}
        ORDER BY market_date
    """, params).fetchall()
    conn.close()

    if not rows:
        print(f"No settled predictions found (last {days} days{', city='+city_filter if city_filter else ''})")
        return

    total = len(rows)
    correct = sum(1 for r in rows if r["model_correct"] == 1)
    clv_vals = [r["clv"] for r in rows if r["clv"] is not None]
    avg_clv = sum(clv_vals) / len(clv_vals) if clv_vals else None

    print(f"\n{'='*60}")
    print(f"CALIBRATION REPORT — last {days} days")
    if city_filter:
        print(f"City filter: {city_filter.upper()}")
    print(f"{'='*60}")
    print(f"Total predictions scored: {total}")
    print(f"Overall accuracy:         {correct}/{total} = {correct/total*100:.1f}%")
    if avg_clv is not None:
        clv_dir = "✓ beating close" if avg_clv > 0 else "✗ worse than close"
        print(f"Avg closing line value:   {avg_clv:+.1f}¢  ({clv_dir})")

    # ── Calibration by probability bucket ────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"{'Prob bucket':<15} {'Predicted':>10} {'Actual win%':>12} {'N':>5} {'Bias':>8}")
    print(f"{'─'*60}")

    buckets = [(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.00)]
    for lo, hi in buckets:
        bucket_rows = [r for r in rows if lo <= r["model_prob"] < hi]
        if not bucket_rows:
            continue
        n = len(bucket_rows)
        wins = sum(1 for r in bucket_rows if r["model_correct"] == 1)
        win_pct = wins / n * 100
        mid = (lo + hi) / 2 * 100
        bias = win_pct - mid
        bias_str = f"{bias:+.1f}%"
        flag = "  ← overconfident" if bias < -5 else ("  ← underconfident" if bias > 5 else "")
        print(f"{int(lo*100)}-{int(hi*100)}%{'':<10} {mid:>9.0f}% {win_pct:>11.1f}% {n:>5}  {bias_str}{flag}")

    # ── Per-city breakdown ────────────────────────────────────────────────────
    from collections import defaultdict
    city_stats: dict = defaultdict(lambda: {"n":0,"correct":0,"clv":[],"ev":[]})
    for r in rows:
        cs = city_stats[r["city_code"]]
        cs["n"] += 1
        cs["correct"] += r["model_correct"] or 0
        if r["clv"] is not None: cs["clv"].append(r["clv"])
        if r["taker_ev"] is not None: cs["ev"].append(r["taker_ev"])

    print(f"\n{'─'*60}")
    print(f"{'City':<20} {'N':>4} {'Acc%':>6} {'Avg EV%':>8} {'Avg CLV':>8}")
    print(f"{'─'*60}")
    for code, cs in sorted(city_stats.items(), key=lambda x: -x[1]["n"]):
        n = cs["n"]
        acc = cs["correct"] / n * 100
        avg_ev = sum(cs["ev"]) / len(cs["ev"]) if cs["ev"] else 0
        avg_c  = sum(cs["clv"]) / len(cs["clv"]) if cs["clv"] else None
        clv_s  = f"{avg_c:+.1f}¢" if avg_c is not None else "  n/a"
        # Flag cities where accuracy is far below model confidence
        avg_prob = sum(r["model_prob"] for r in rows if r["city_code"] == code) / n
        gap = (cs["correct"] / n) - avg_prob
        flag = "  ← recalibrate bias" if gap < -0.08 else ""
        print(f"{code:<20} {n:>4} {acc:>5.1f}% {avg_ev:>7.1f}% {clv_s:>8}{flag}")

    # ── Recent form (last 10) ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Recent predictions (last 10):")
    print(f"{'─'*60}")
    for r in list(rows)[-10:]:
        status = "✓" if r["model_correct"] else "✗"
        clv_s  = f"CLV {r['clv']:+.0f}¢" if r["clv"] is not None else ""
        print(f"  {status} {r['market_date']} {r['city_code']:<6} "
              f"model={r['model_prob']*100:.0f}% side={r['best_side'] or '?'} "
              f"ev={r['taker_ev'] or 0:.1f}%  {clv_s}")

    print(f"{'='*60}\n")

# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] == "report":
        city  = None
        days  = 90
        for a in args[1:]:
            if a.startswith("--days="):
                days = int(a.split("=")[1])
            elif a == "--days" and args.index(a)+1 < len(args):
                days = int(args[args.index(a)+1])
            elif not a.startswith("--"):
                city = a
        calibration_report(city_filter=city, days=days)

    elif args[0] == "score":
        target = None
        if len(args) > 1:
            target = date.fromisoformat(args[1])
        score_settlements(target)

    elif args[0] == "summary":
        # Quick one-liner: total logged, total scored, accuracy
        conn = get_db()
        total   = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        scored  = conn.execute("SELECT COUNT(*) FROM predictions WHERE settled=1").fetchone()[0]
        correct = conn.execute("SELECT COUNT(*) FROM predictions WHERE model_correct=1").fetchone()[0]
        pending = total - scored
        conn.close()
        print(f"Total logged: {total} | Scored: {scored} | Pending: {pending}")
        if scored:
            print(f"Accuracy: {correct}/{scored} = {correct/scored*100:.1f}%")

    else:
        print(__doc__)
