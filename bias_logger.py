"""
bias_logger.py — Kalshi Weather Bot prediction tracker (PostgreSQL version)

Switched from SQLite to PostgreSQL so both the bot service and cron scorer
service can share the same database over the network on Railway.

Requires: pip install psycopg2-binary

Three jobs:
  1. log_prediction()     — called by the bot on every alert
  2. score_settlements()  — run by cron nightly, fetches Kalshi results
  3. calibration_report() — prints accuracy + CLV table

Usage:
  python3 bias_logger.py score           # score yesterday's markets
  python3 bias_logger.py score 2026-05-25  # score a specific date
  python3 bias_logger.py report          # full calibration report
  python3 bias_logger.py report NYC      # report for one city
  python3 bias_logger.py report --days 30
  python3 bias_logger.py summary         # quick one-liner
"""

import os, sys, requests
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

DATABASE_URL = os.environ.get("DATABASE_URL", "")
ET_TZ        = ZoneInfo("America/New_York")
KALSHI_BASE  = "https://external-api.kalshi.com/trade-api/v2"

# ── DB CONNECTION ─────────────────────────────────────────────────────────────
def get_conn():
    import psycopg2
    return psycopg2.connect(DATABASE_URL)

# ── SCHEMA ────────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    logged_at       TEXT NOT NULL,
    market_date     TEXT NOT NULL,
    city_code       TEXT NOT NULL,
    city_name       TEXT NOT NULL,
    ticker          TEXT NOT NULL UNIQUE,
    threshold_f     REAL NOT NULL,
    threshold_kind  TEXT NOT NULL,
    model_prob      REAL NOT NULL,
    yes_price       INTEGER NOT NULL,
    no_price        INTEGER NOT NULL,
    best_side       TEXT,
    taker_ev        REAL,
    ensemble_mean   REAL,
    spread          REAL,
    confidence      TEXT,
    bias_applied    REAL,
    asos_obs_high   REAL,
    ecmwf_high      REAL,
    nbm_high        REAL,
    hrrr_high       REAL,
    icon_high       REAL,
    settled         INTEGER DEFAULT 0,
    actual_high_f   REAL,
    yes_result      INTEGER,
    model_correct   INTEGER,
    closing_yes_price INTEGER,
    clv             REAL
);
CREATE INDEX IF NOT EXISTS idx_market_date ON predictions(market_date);
CREATE INDEX IF NOT EXISTS idx_city        ON predictions(city_code);
CREATE INDEX IF NOT EXISTS idx_settled     ON predictions(settled);
"""

def ensure_schema():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(SCHEMA)
        conn.commit()
    finally:
        conn.close()

# ── LOG PREDICTION ────────────────────────────────────────────────────────────
def log_prediction(
    market_date, city_code, city_name, ticker, threshold_f,
    forecast, model_prob, yes_price, no_price, asos_obs_high,
    best_side=None, taker_ev=None, threshold_kind="T",
):
    ensure_schema()
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO predictions (
                logged_at, market_date, city_code, city_name, ticker,
                threshold_f, threshold_kind, model_prob, yes_price, no_price,
                best_side, taker_ev, ensemble_mean, spread, confidence,
                bias_applied, asos_obs_high, ecmwf_high, nbm_high, hrrr_high, icon_high
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (ticker) DO NOTHING
        """, (
            datetime.now(ET_TZ).isoformat(),
            str(market_date), city_code, city_name, ticker,
            threshold_f, threshold_kind, model_prob, yes_price, no_price,
            best_side, taker_ev,
            forecast.get("ensemble_mean"), forecast.get("spread"),
            forecast.get("confidence"),   forecast.get("bias_applied"),
            asos_obs_high,
            forecast.get("ecmwf_high"),   forecast.get("nbm_high"),
            forecast.get("hrrr_high"),    forecast.get("icon_high"),
        ))
        conn.commit()
    except Exception as e:
        print(f"[bias_logger] log error: {e}")
    finally:
        conn.close()

# ── SETTLEMENT SCORING ────────────────────────────────────────────────────────
def fetch_kalshi_result(ticker):
    try:
        r = requests.get(f"{KALSHI_BASE}/markets/{ticker}", timeout=10)
        r.raise_for_status()
        m = r.json().get("market", {})
        if m.get("status") not in ("settled", "finalized"):
            return None
        result = m.get("result", "")
        yes_result = 1 if result == "yes" else (0 if result == "no" else None)
        if yes_result is None:
            return None
        closing = m.get("last_price")
        closing_cents = round(float(closing) * 100) if closing else None
        return {"yes_result": yes_result, "closing_yes_price": closing_cents}
    except Exception as e:
        print(f"[bias_logger] fetch result {ticker}: {e}")
        return None

def score_settlements(target_date=None):
    ensure_schema()
    if target_date is None:
        target_date = date.today() - timedelta(days=1)
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, ticker, best_side, yes_price, model_prob
            FROM predictions
            WHERE settled = 0 AND market_date <= %s
            ORDER BY market_date
        """, (str(target_date),))
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        print(f"[scorer] No unscored predictions for {target_date} or earlier")
        return

    print(f"[scorer] Scoring {len(rows)} predictions...")
    scored = 0

    for row in rows:
        rid, ticker, best_side, yes_price, model_prob = row
        result = fetch_kalshi_result(ticker)
        if result is None:
            print(f"  ⏳ {ticker}: not settled yet")
            continue

        yes_result    = result["yes_result"]
        closing_price = result["closing_yes_price"]

        if best_side == "YES":
            model_correct = 1 if yes_result == 1 else 0
            entry_price   = yes_price
        elif best_side == "NO":
            model_correct = 1 if yes_result == 0 else 0
            entry_price   = 100 - yes_price
        else:
            model_correct = None
            entry_price   = None

        clv = None
        if closing_price is not None and entry_price is not None:
            if best_side == "YES":
                clv = round(closing_price - entry_price, 1)
            else:
                clv = round((100 - closing_price) - entry_price, 1)

        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute("""
                UPDATE predictions
                SET settled=1, yes_result=%s, model_correct=%s,
                    closing_yes_price=%s, clv=%s
                WHERE id=%s
            """, (yes_result, model_correct, closing_price, clv, rid))
            conn.commit()
        finally:
            conn.close()

        status = "✓" if model_correct else "✗"
        clv_str = f"CLV {clv:+.0f}¢" if clv is not None else "no CLV"
        print(f"  {status} {ticker}: {'YES' if yes_result else 'NO'} won | {clv_str}")
        scored += 1

    print(f"[scorer] Scored {scored}/{len(rows)} predictions")

# ── CALIBRATION REPORT ────────────────────────────────────────────────────────
def calibration_report(city_filter=None, days=90):
    ensure_schema()
    cutoff = str(date.today() - timedelta(days=days))
    conn = get_conn()
    try:
        cur = conn.cursor()
        query = """
            SELECT city_code, city_name, model_prob, best_side, taker_ev,
                   model_correct, clv, ensemble_mean, spread, market_date, ticker
            FROM predictions
            WHERE settled = 1 AND market_date >= %s
        """
        params = [cutoff]
        if city_filter:
            query += " AND city_code = %s"
            params.append(city_filter.upper())
        query += " ORDER BY market_date"
        cur.execute(query, params)
        rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        print(f"No settled predictions (last {days} days)")
        return

    total   = len(rows)
    correct = sum(1 for r in rows if r[5] == 1)
    clv_vals = [r[6] for r in rows if r[6] is not None]
    avg_clv  = sum(clv_vals) / len(clv_vals) if clv_vals else None

    print(f"\n{'='*60}")
    print(f"CALIBRATION REPORT — last {days} days")
    if city_filter: print(f"City: {city_filter.upper()}")
    print(f"{'='*60}")
    print(f"Total scored: {total}")
    print(f"Accuracy:     {correct}/{total} = {correct/total*100:.1f}%")
    if avg_clv is not None:
        flag = "✓ beating close" if avg_clv > 0 else "✗ worse than close"
        print(f"Avg CLV:      {avg_clv:+.1f}¢  ({flag})")

    print(f"\n{'─'*60}")
    print(f"{'Prob bucket':<15} {'Predicted':>10} {'Actual%':>8} {'N':>5} {'Bias':>8}")
    print(f"{'─'*60}")
    for lo, hi in [(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.00)]:
        bucket = [r for r in rows if lo <= r[2] < hi]
        if not bucket: continue
        n = len(bucket)
        wins = sum(1 for r in bucket if r[5] == 1)
        win_pct = wins / n * 100
        mid = (lo + hi) / 2 * 100
        bias = win_pct - mid
        flag = "  ← overconfident" if bias < -5 else ("  ← underconfident" if bias > 5 else "")
        print(f"{int(lo*100)}-{int(hi*100)}%{'':<10} {mid:>9.0f}% {win_pct:>7.1f}% {n:>5}  {bias:+.1f}%{flag}")

    from collections import defaultdict
    city_stats = defaultdict(lambda: {"n":0,"correct":0,"clv":[],"ev":[]})
    for r in rows:
        cs = city_stats[r[0]]
        cs["n"] += 1
        cs["correct"] += r[5] or 0
        if r[6] is not None: cs["clv"].append(r[6])
        if r[4] is not None: cs["ev"].append(r[4])

    print(f"\n{'─'*60}")
    print(f"{'City':<20} {'N':>4} {'Acc%':>6} {'Avg EV%':>8} {'Avg CLV':>8}")
    print(f"{'─'*60}")
    for code, cs in sorted(city_stats.items(), key=lambda x: -x[1]["n"]):
        n      = cs["n"]
        acc    = cs["correct"] / n * 100
        avg_ev = sum(cs["ev"]) / len(cs["ev"]) if cs["ev"] else 0
        avg_c  = sum(cs["clv"]) / len(cs["clv"]) if cs["clv"] else None
        clv_s  = f"{avg_c:+.1f}¢" if avg_c is not None else "  n/a"
        avg_prob = sum(r[2] for r in rows if r[0] == code) / n
        gap = (cs["correct"] / n) - avg_prob
        flag = "  ← recalibrate" if gap < -0.08 else ""
        print(f"{code:<20} {n:>4} {acc:>5.1f}% {avg_ev:>7.1f}% {clv_s:>8}{flag}")

    print(f"\n{'─'*60}")
    print("Recent predictions (last 10):")
    for r in list(rows)[-10:]:
        status = "✓" if r[5] else "✗"
        clv_s  = f"CLV {r[6]:+.0f}¢" if r[6] is not None else ""
        print(f"  {status} {r[9]} {r[0]:<6} model={r[2]*100:.0f}% "
              f"side={r[3] or '?'} ev={r[4] or 0:.1f}%  {clv_s}")
    print(f"{'='*60}\n")

# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] == "report":
        city, days = None, 90
        for a in args[1:]:
            if a.startswith("--days="):   days = int(a.split("=")[1])
            elif a == "--days" and args.index(a)+1 < len(args):
                days = int(args[args.index(a)+1])
            elif not a.startswith("--"):  city = a
        calibration_report(city_filter=city, days=days)

    elif args[0] == "score":
        target = date.fromisoformat(args[1]) if len(args) > 1 else None
        score_settlements(target)

    elif args[0] == "summary":
        ensure_schema()
        conn = get_conn()
        cur  = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM predictions")
        total = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM predictions WHERE settled=1")
        scored = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM predictions WHERE model_correct=1")
        correct = cur.fetchone()[0]
        conn.close()
        print(f"Total logged: {total} | Scored: {scored} | Pending: {total-scored}")
        if scored:
            print(f"Accuracy: {correct}/{scored} = {correct/scored*100:.1f}%")
    else:
        print(__doc__)
