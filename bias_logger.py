"""
bias_logger.py — Kalshi Weather Bot prediction tracker (PostgreSQL version)

Switched from SQLite to PostgreSQL so both the bot service and cron scorer
service can share the same database over the network on Railway.

Requires: pip install psycopg2-binary

Three jobs:
  1. log_prediction()     — called by the bot on every alert
  2. score_settlements()  — run by cron nightly, fetches Kalshi results
  3. calibration_report() — prints accuracy + CLV table

v3.27: Added bet_category (overnight/morning/pacing) and actual_high_f storage
       so the scoreboard can break down results by bet type and show final temps.
v3.38: Fixed calibration_report date filter — the ::date cast on market_date
       was silently excluding every row whose stored value didn't cast cleanly
       (time component / odd format), so reports returned "No settled
       predictions" even with scored rows present. Switched to a plain string
       comparison on LEFT(market_date,10), which sorts correctly for YYYY-MM-DD.
       Added a 'cal' subcommand: a quoting-free calibration dump (overall,
       prob-bucket calibration, by-side, by-category) that runs cleanly as a
       Railway start command with no multi-line / nested-quote issues.
v3.39: (1) log_prediction now UPSERTS instead of ON CONFLICT DO NOTHING, so a
       later same-day scan (e.g. afternoon pace-confirmed) overwrites the
       earlier row instead of being silently dropped. Only overwrites while the
       row is unsettled, so it never clobbers a scored result.
       (2) Added orderbook depth columns (book_best_price, book_liq_at_best,
       book_total_liq, book_levels) + log_orderbook() so the bot can record the
       live book at alert time. This lets the slippage/maker analysis use real
       depth instead of a flat haircut.

Usage:
  python3 bias_logger.py score             # score yesterday's markets
  python3 bias_logger.py score 2026-05-25  # score a specific date
  python3 bias_logger.py report            # full calibration report
  python3 bias_logger.py report NYC        # report for one city
  python3 bias_logger.py report --days 30
  python3 bias_logger.py cal               # quick calibration dump (start-command friendly)
  python3 bias_logger.py cal --days 30     # calibration dump, last 30 days
  python3 bias_logger.py summary           # quick one-liner
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
    bet_category    TEXT DEFAULT 'morning',
    book_best_price   INTEGER,
    book_liq_at_best  REAL,
    book_total_liq    REAL,
    book_levels       TEXT,
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

# Columns added after initial deploy — added defensively via ALTER
MIGRATIONS = [
    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS bet_category TEXT DEFAULT 'morning';",
    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS actual_high_f REAL;",
    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS book_best_price INTEGER;",
    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS book_liq_at_best REAL;",
    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS book_total_liq REAL;",
    "ALTER TABLE predictions ADD COLUMN IF NOT EXISTS book_levels TEXT;",
]

def ensure_schema():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(SCHEMA)
        for migration in MIGRATIONS:
            try:
                cur.execute(migration)
            except Exception as e:
                print(f"[schema] migration skipped: {e}")
        conn.commit()
    finally:
        conn.close()

# ── LOG PREDICTION ────────────────────────────────────────────────────────────
def log_prediction(
    market_date, city_code, city_name, ticker, threshold_f,
    forecast, model_prob, yes_price, no_price, asos_obs_high,
    best_side=None, taker_ev=None, threshold_kind="T",
    bet_category="morning",
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
                bias_applied, asos_obs_high, ecmwf_high, nbm_high, hrrr_high,
                icon_high, bet_category
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (ticker) DO UPDATE SET
                logged_at      = EXCLUDED.logged_at,
                market_date    = EXCLUDED.market_date,
                city_code      = EXCLUDED.city_code,
                city_name      = EXCLUDED.city_name,
                threshold_f    = EXCLUDED.threshold_f,
                threshold_kind = EXCLUDED.threshold_kind,
                model_prob     = EXCLUDED.model_prob,
                yes_price      = EXCLUDED.yes_price,
                no_price       = EXCLUDED.no_price,
                best_side      = EXCLUDED.best_side,
                taker_ev       = EXCLUDED.taker_ev,
                ensemble_mean  = EXCLUDED.ensemble_mean,
                spread         = EXCLUDED.spread,
                confidence     = EXCLUDED.confidence,
                bias_applied   = EXCLUDED.bias_applied,
                asos_obs_high  = EXCLUDED.asos_obs_high,
                ecmwf_high     = EXCLUDED.ecmwf_high,
                nbm_high       = EXCLUDED.nbm_high,
                hrrr_high      = EXCLUDED.hrrr_high,
                icon_high      = EXCLUDED.icon_high,
                bet_category   = EXCLUDED.bet_category
            WHERE predictions.settled = 0
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
            bet_category,
        ))
        conn.commit()
    except Exception as e:
        print(f"[bias_logger] log error: {e}")
    finally:
        conn.close()

# ── LOG ORDERBOOK DEPTH ───────────────────────────────────────────────────────
def log_orderbook(ticker, liq, levels=None):
    """Attach orderbook depth (captured at alert time) to an existing row.
    Never creates rows; updates by ticker only while the row is unsettled.
    `liq` is the dict from fetch_orderbook_liquidity; `levels` is the take
    ladder for our side: [[take_price_cents, qty], ...]."""
    if not ticker or liq is None:
        return
    import json
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            UPDATE predictions
            SET book_best_price  = %s,
                book_liq_at_best = %s,
                book_total_liq   = %s,
                book_levels      = %s
            WHERE ticker = %s AND settled = 0
        """, (
            liq.get("best_price_cents"),
            liq.get("liq_at_best"),
            liq.get("total_liq"),
            json.dumps(levels) if levels is not None else None,
            ticker,
        ))
        conn.commit()
    except Exception as e:
        print(f"[bias_logger] orderbook log error: {e}")
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

        # Actual high temp from expiration_value
        actual_high = None
        exp_val = m.get("expiration_value")
        if exp_val not in (None, ""):
            try:
                actual_high = float(exp_val)
            except (ValueError, TypeError):
                actual_high = None

        # Try close_price first, then yes_bid, then last_price as fallback
        closing = (
            m.get("close_price") or
            m.get("yes_bid") or
            m.get("last_price")
        )
        closing_cents = round(float(closing) * 100) if closing else None

        return {
            "yes_result":        yes_result,
            "closing_yes_price": closing_cents,
            "actual_high_f":     actual_high,
        }
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
        actual_high   = result["actual_high_f"]

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
                    closing_yes_price=%s, clv=%s, actual_high_f=%s
                WHERE id=%s
            """, (yes_result, model_correct, closing_price, clv, actual_high, rid))
            conn.commit()
        finally:
            conn.close()

        status = "✓" if model_correct else "✗"
        clv_str = f"CLV {clv:+.0f}¢" if clv is not None else "no CLV"
        temp_str = f"{actual_high:.0f}°F" if actual_high is not None else "?°F"
        print(f"  {status} {ticker}: {'YES' if yes_result else 'NO'} won @ {temp_str} | {clv_str}")
        scored += 1

    print(f"[scorer] Scored {scored}/{len(rows)} predictions")

# ── CALIBRATION REPORT ────────────────────────────────────────────────────────
def calibration_report(city_filter=None, days=90):
    ensure_schema()
    cutoff = str(date.today() - timedelta(days=days))
    conn = get_conn()
    try:
        cur = conn.cursor()
        # market_date is TEXT. A previous version cast LEFT(market_date,10) to
        # ::date for the comparison, but any row whose stored value didn't cast
        # cleanly (time component, blank, odd format) was silently dropped —
        # which made the whole report return "No settled predictions" even when
        # scored rows existed. YYYY-MM-DD strings sort correctly lexically, so a
        # plain string comparison is both correct and robust here.
        query = """
            SELECT city_code, city_name, model_prob, best_side, taker_ev,
                   model_correct, clv, ensemble_mean, spread, market_date, ticker
            FROM predictions
            WHERE settled = 1
              AND LEFT(market_date, 10) >= %s
        """
        params = [cutoff]
        if city_filter:
            query += " AND city_code = %s"
            params.append(city_filter.upper())
        query += " ORDER BY LEFT(market_date, 10)"
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

    # ── BY CATEGORY ───────────────────────────────────────────────────────────
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT bet_category, COUNT(*),
                   SUM(CASE WHEN model_correct=1 THEN 1 ELSE 0 END)
            FROM predictions
            WHERE settled=1
              AND LEFT(market_date,10) >= %s
            GROUP BY bet_category
        """, (cutoff,))
        cat_rows = cur.fetchall()
    finally:
        conn.close()

    if cat_rows:
        print(f"\n{'─'*60}")
        print(f"{'Category':<16} {'N':>4} {'Acc%':>6}")
        print(f"{'─'*60}")
        for cat, n, c in sorted(cat_rows, key=lambda x: -(x[1] or 0)):
            acc = (c or 0) / n * 100 if n else 0
            print(f"{(cat or 'none'):<16} {n:>4} {acc:>5.1f}%")

    # ── BY SIDE (NO vs YES) ───────────────────────────────────────────────────
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT best_side, COUNT(*),
                   SUM(CASE WHEN model_correct=1 THEN 1 ELSE 0 END)
            FROM predictions
            WHERE settled=1
              AND LEFT(market_date,10) >= %s
            GROUP BY best_side
        """, (cutoff,))
        side_rows = cur.fetchall()
    finally:
        conn.close()

    if side_rows:
        print(f"\n{'─'*60}")
        print(f"{'Side':<16} {'N':>4} {'Acc%':>6}")
        print(f"{'─'*60}")
        for sd, n, c in side_rows:
            acc = (c or 0) / n * 100 if n else 0
            print(f"{(sd or 'none'):<16} {n:>4} {acc:>5.1f}%")

    print(f"{'='*60}\n")

# ── QUICK CALIBRATION DUMP (start-command friendly) ───────────────────────────
def cal_dump(days=90):
    """
    Quoting-free calibration dump designed to run cleanly as a Railway start
    command (no multi-line python -c, no nested quotes). Prints overall
    accuracy, the prob-bucket calibration table, and by-side / by-category
    breakdowns. Uses the same robust LEFT(market_date,10) string comparison as
    calibration_report so it never silently drops rows.
    """
    ensure_schema()
    cutoff = str(date.today() - timedelta(days=days))
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT model_prob, model_correct, best_side, bet_category,
                   threshold_kind, taker_ev
            FROM predictions
            WHERE settled = 1
              AND model_correct IS NOT NULL
              AND LEFT(market_date, 10) >= %s
        """, (cutoff,))
        rows = cur.fetchall()
    finally:
        conn.close()

    print(f"TOTAL settled (last {days}d): {len(rows)}")
    if not rows:
        return

    corr = sum(r[1] for r in rows)
    print(f"Overall: {corr}/{len(rows)} = {corr/len(rows)*100:.1f}%")

    print("--- calibration (predicted prob vs actual win%) ---")
    for lo, hi in [(0.0,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.01)]:
        b = [r for r in rows if lo <= r[0] < hi]
        if not b:
            continue
        n = len(b)
        won = sum(x[1] for x in b) / n * 100
        mid = (lo + min(hi, 1.0)) / 2 * 100
        gap = won - mid
        flag = ""
        if gap < -7:  flag = "  <- overconfident"
        elif gap > 7: flag = "  <- underconfident"
        print(f"  {int(lo*100):>3}-{int(min(hi,1.0)*100):<3}% | n={n:>3} | won {won:5.1f}% (mid {mid:4.0f}%, gap {gap:+5.1f}%){flag}")

    print("--- by side ---")
    for s in sorted(set(r[2] for r in rows), key=lambda x: x or ""):
        b = [r for r in rows if r[2] == s]
        n = len(b); won = sum(x[1] for x in b) / n * 100
        print(f"  {str(s or 'none'):<6} n={n:>3} won {won:5.1f}%")

    print("--- by category ---")
    for c in sorted(set(r[3] for r in rows), key=lambda x: x or ""):
        b = [r for r in rows if r[3] == c]
        n = len(b); won = sum(x[1] for x in b) / n * 100
        print(f"  {str(c or 'none'):<14} n={n:>3} won {won:5.1f}%")

    print("--- by threshold kind ---")
    for k in sorted(set(r[4] for r in rows), key=lambda x: x or ""):
        b = [r for r in rows if r[4] == k]
        n = len(b); won = sum(x[1] for x in b) / n * 100
        print(f"  {str(k or 'none'):<6} n={n:>3} won {won:5.1f}%")

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

    elif args[0] == "cal":
        days = 90
        for a in args[1:]:
            if a.startswith("--days="): days = int(a.split("=")[1])
            elif a == "--days" and args.index(a)+1 < len(args):
                days = int(args[args.index(a)+1])
        cal_dump(days=days)

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
