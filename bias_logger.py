"""
bias_logger.py — Kalshi Weather Bot Prediction Tracker

Three jobs:
  1. log_prediction()  — called by main bot on every alert candidate
  2. score()           — called nightly by cron; fetches Kalshi results and marks won/lost
  3. report()          — prints calibration summary; run manually anytime

DB path: /data/predictions.db  (Railway Volume mount)
         Falls back to ./predictions.db if /data doesn't exist.
"""

import os
import sqlite3
import requests
import json
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

# ── DB PATH ───────────────────────────────────────────────────────────────────
# Railway: mount a Volume at /data so the DB survives redeploys.
# Locally or without a Volume: falls back to current directory.
_DATA_DIR = "/data" if os.path.isdir("/data") else "."
DB_PATH   = os.path.join(_DATA_DIR, "predictions.db")

ET_TZ         = ZoneInfo("America/New_York")
KALSHI_BASE   = "https://external-api.kalshi.com/trade-api/v2"
KALSHI_KEY    = os.environ.get("KALSHI_API_KEY", "")   # set in Railway env vars
KALSHI_EMAIL  = os.environ.get("KALSHI_EMAIL", "")
KALSHI_PASS   = os.environ.get("KALSHI_PASSWORD", "")

# ── SCHEMA ────────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    pred_date       TEXT    NOT NULL,          -- YYYY-MM-DD market date
    logged_at       TEXT    NOT NULL,          -- ISO timestamp when logged
    city_code       TEXT    NOT NULL,
    city_name       TEXT    NOT NULL,
    ticker          TEXT    NOT NULL,
    threshold_f     REAL    NOT NULL,
    threshold_kind  TEXT    NOT NULL,          -- T or B
    best_side       TEXT    NOT NULL,          -- YES or NO
    model_prob      REAL    NOT NULL,          -- 0-100
    yes_price       INTEGER NOT NULL,          -- cents
    no_price        INTEGER NOT NULL,          -- cents
    taker_ev        REAL    NOT NULL,          -- percent
    ensemble_mean   REAL,
    corrected_mean  REAL,
    spread          REAL,
    bias_applied    REAL,
    confidence      TEXT,
    obs_high        REAL,
    result_high     REAL,                      -- actual observed high (filled by scorer)
    won             INTEGER,                   -- 1=win 0=loss NULL=unscored
    clv             REAL,                      -- closing line value (filled by scorer)
    closing_price   INTEGER,                   -- final Kalshi price before settlement
    scored_at       TEXT,                      -- ISO timestamp when scored
    UNIQUE(ticker, best_side, logged_at)       -- prevent duplicate log rows
);

CREATE TABLE IF NOT EXISTS daily_summary (
    summary_date    TEXT PRIMARY KEY,
    total_alerts    INTEGER,
    fire_alerts     INTEGER,
    watch_alerts    INTEGER,
    scored          INTEGER,
    wins            INTEGER,
    losses          INTEGER,
    win_rate        REAL,
    avg_ev          REAL,
    avg_clv         REAL,
    created_at      TEXT
);
"""

def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = _get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()

# ── LOG PREDICTION ────────────────────────────────────────────────────────────
def log_prediction(
    pred_date,          # date object
    city_code: str,
    city_name: str,
    ticker: str,
    threshold_f: float,
    forecast: dict,
    model_prob: float,  # 0-1 float
    yes_price: int,
    no_price: int,
    obs_high,           # float or None
    best_side: str = "YES",
    taker_ev: float = 0.0,
    threshold_kind: str = "T",
):
    """
    Called by the main bot every time it evaluates a market.
    Silently skips duplicates (same ticker + side logged within same minute).
    """
    init_db()
    logged_at = datetime.now(ET_TZ).isoformat()
    # Deduplicate within the same minute to avoid log spam from rescans
    minute_ts = logged_at[:16]  # YYYY-MM-DDTHH:MM

    conn = _get_conn()
    try:
        # Check if we already logged this ticker+side in the same minute
        existing = conn.execute(
            "SELECT id FROM predictions WHERE ticker=? AND best_side=? AND logged_at LIKE ?",
            (ticker, best_side, f"{minute_ts}%")
        ).fetchone()
        if existing:
            return  # already logged this minute, skip

        conn.execute("""
            INSERT OR IGNORE INTO predictions
            (pred_date, logged_at, city_code, city_name, ticker,
             threshold_f, threshold_kind, best_side, model_prob,
             yes_price, no_price, taker_ev,
             ensemble_mean, corrected_mean, spread, bias_applied,
             confidence, obs_high)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            str(pred_date),
            logged_at,
            city_code,
            city_name,
            ticker,
            threshold_f,
            threshold_kind,
            best_side,
            round(model_prob * 100, 2),
            yes_price,
            no_price,
            taker_ev,
            forecast.get("ensemble_mean"),
            forecast.get("corrected_mean"),
            forecast.get("spread"),
            forecast.get("bias_applied"),
            forecast.get("confidence"),
            obs_high,
        ))
        conn.commit()
        print(f"[bias_logger] logged {ticker} {best_side} EV={taker_ev}%")
    except Exception as e:
        print(f"[bias_logger] log error: {e}")
    finally:
        conn.close()

# ── KALSHI AUTH ───────────────────────────────────────────────────────────────
_kalshi_token = None

def _get_kalshi_token():
    global _kalshi_token
    if _kalshi_token:
        return _kalshi_token
    if not KALSHI_EMAIL or not KALSHI_PASS:
        print("[scorer] KALSHI_EMAIL / KALSHI_PASSWORD not set — cannot fetch results")
        return None
    try:
        r = requests.post(f"{KALSHI_BASE}/login",
            json={"email": KALSHI_EMAIL, "password": KALSHI_PASS}, timeout=10)
        r.raise_for_status()
        _kalshi_token = r.json().get("token")
        return _kalshi_token
    except Exception as e:
        print(f"[scorer] Kalshi auth failed: {e}")
        return None

def _kalshi_headers():
    token = _get_kalshi_token()
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}

# ── FETCH MARKET RESULT ───────────────────────────────────────────────────────
def _fetch_market_result(ticker: str) -> dict | None:
    """
    Returns dict with keys: result ('yes'/'no'/''), closing_yes_price
    """
    try:
        r = requests.get(f"{KALSHI_BASE}/markets/{ticker}",
            headers=_kalshi_headers(), timeout=10)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        m = r.json().get("market", {})
        result        = m.get("result", "")          # 'yes', 'no', or ''
        closing_price = m.get("last_price")          # last traded price in cents
        return {"result": result, "closing_price": closing_price}
    except Exception as e:
        print(f"[scorer] fetch {ticker}: {e}")
        return None

# ── SCORE ─────────────────────────────────────────────────────────────────────
def score(target_date: date | None = None):
    """
    Fetch results from Kalshi for all unscored predictions on target_date
    (defaults to yesterday ET). Mark each as won=1 or won=0.
    Also computes CLV = closing_price - entry_price for the side we took.

    Run via cron: `python bias_logger.py score`
    """
    init_db()
    if target_date is None:
        target_date = (datetime.now(ET_TZ) - timedelta(days=1)).date()

    ds = str(target_date)
    print(f"[scorer] Scoring predictions for {ds}")

    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE pred_date=? AND won IS NULL",
        (ds,)
    ).fetchall()
    conn.close()

    if not rows:
        print(f"[scorer] No unscored predictions for {ds}")
        return

    print(f"[scorer] Found {len(rows)} unscored predictions")
    scored = wins = losses = 0

    for row in rows:
        ticker    = row["ticker"]
        best_side = row["best_side"]
        entry_yes = row["yes_price"]
        entry_no  = row["no_price"]

        info = _fetch_market_result(ticker)
        if info is None:
            print(f"[scorer] {ticker}: not found on Kalshi, skipping")
            continue

        result        = info["result"]          # 'yes' or 'no'
        closing_price = info["closing_price"]   # cents

        if result not in ("yes", "no"):
            print(f"[scorer] {ticker}: not yet settled (result='{result}'), skipping")
            continue

        # Did our side win?
        won = 1 if result.upper() == best_side else 0

        # CLV: closing price of our side vs our entry price
        # Positive CLV = we got better price than where market closed
        if best_side == "YES":
            entry_price   = entry_yes
            closing_side  = closing_price if closing_price else entry_yes
        else:
            entry_price   = entry_no
            # NO price = 100 - YES price
            closing_side  = (100 - closing_price) if closing_price else entry_no

        clv = closing_side - entry_price  # positive = we beat the close

        conn = _get_conn()
        conn.execute("""
            UPDATE predictions
            SET won=?, clv=?, closing_price=?, scored_at=?
            WHERE id=?
        """, (won, clv, closing_price, datetime.now(ET_TZ).isoformat(), row["id"]))
        conn.commit()
        conn.close()

        status = "✅ WIN" if won else "❌ LOSS"
        print(f"[scorer] {ticker} {best_side}: {status} | CLV={clv:+d}¢")
        scored += 1
        if won:
            wins += 1
        else:
            losses += 1

    win_rate = round(wins / scored * 100, 1) if scored else 0
    print(f"\n[scorer] Done: {scored} scored | {wins}W {losses}L | Win rate: {win_rate}%")

    # Write daily summary
    _write_daily_summary(ds)

def _write_daily_summary(ds: str):
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE pred_date=? AND won IS NOT NULL", (ds,)
    ).fetchall()
    all_rows = conn.execute(
        "SELECT * FROM predictions WHERE pred_date=?", (ds,)
    ).fetchall()

    total   = len(all_rows)
    scored  = len(rows)
    wins    = sum(1 for r in rows if r["won"] == 1)
    losses  = sum(1 for r in rows if r["won"] == 0)
    wr      = round(wins / scored * 100, 1) if scored else 0
    avg_ev  = round(sum(r["taker_ev"] for r in all_rows) / total, 1) if total else 0
    avg_clv = round(sum(r["clv"] for r in rows if r["clv"] is not None) / scored, 1) if scored else 0

    conn.execute("""
        INSERT OR REPLACE INTO daily_summary
        (summary_date, total_alerts, fire_alerts, watch_alerts, scored,
         wins, losses, win_rate, avg_ev, avg_clv, created_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, (ds, total, 0, 0, scored, wins, losses, wr, avg_ev, avg_clv,
          datetime.now(ET_TZ).isoformat()))
    conn.commit()
    conn.close()

# ── REPORT ────────────────────────────────────────────────────────────────────
def report(days: int = 7):
    """
    Print calibration report for the last N days.
    Run manually: `python bias_logger.py report`
    """
    init_db()
    since = str((datetime.now(ET_TZ) - timedelta(days=days)).date())

    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM predictions WHERE pred_date >= ? AND won IS NOT NULL ORDER BY pred_date",
        (since,)
    ).fetchall()
    conn.close()

    if not rows:
        print(f"[report] No scored predictions in the last {days} days.")
        return

    total  = len(rows)
    wins   = sum(1 for r in rows if r["won"] == 1)
    losses = total - wins
    wr     = round(wins / total * 100, 1)
    avg_ev = round(sum(r["taker_ev"] for r in rows) / total, 1)
    clv_rows = [r["clv"] for r in rows if r["clv"] is not None]
    avg_clv  = round(sum(clv_rows) / len(clv_rows), 1) if clv_rows else 0

    print(f"\n{'='*55}")
    print(f"  BIAS CALIBRATION REPORT  (last {days} days)")
    print(f"{'='*55}")
    print(f"  Total scored : {total}  ({wins}W / {losses}L)")
    print(f"  Win rate     : {wr}%")
    print(f"  Avg EV       : +{avg_ev}%")
    print(f"  Avg CLV      : {avg_clv:+.1f}¢  ({'✅ positive' if avg_clv >= 0 else '❌ negative'})")

    # ── Per-city breakdown ────────────────────────────────────────────────────
    print(f"\n  {'CITY':<18} {'BETS':>5} {'WIN%':>6} {'AVG EV':>8} {'AVG CLV':>8}  FLAG")
    print(f"  {'-'*58}")
    cities = sorted(set(r["city_code"] for r in rows))
    for cc in cities:
        cr    = [r for r in rows if r["city_code"] == cc]
        cw    = sum(1 for r in cr if r["won"] == 1)
        cwr   = round(cw / len(cr) * 100, 1)
        cev   = round(sum(r["taker_ev"] for r in cr) / len(cr), 1)
        cclv  = [r["clv"] for r in cr if r["clv"] is not None]
        cavg  = round(sum(cclv) / len(cclv), 1) if cclv else 0
        name  = cr[0]["city_name"][:16]
        flag  = ""
        if len(cr) >= 3:
            if cwr < 45:   flag = "← recalibrate bias (too low)"
            elif cwr > 75: flag = "← recalibrate bias (too high)"
            if cavg < -5:  flag = flag or "← negative CLV"
        print(f"  {name:<18} {len(cr):>5} {cwr:>5.1f}% {cev:>+7.1f}% {cavg:>+7.1f}¢  {flag}")

    # ── Confidence bucket breakdown ───────────────────────────────────────────
    print(f"\n  {'CONFIDENCE':<12} {'BETS':>5} {'WIN%':>6}  (model calibration)")
    print(f"  {'-'*35}")
    for conf in ("high", "medium", "low"):
        cr  = [r for r in rows if r["confidence"] == conf]
        if not cr: continue
        cw  = sum(1 for r in cr if r["won"] == 1)
        cwr = round(cw / len(cr) * 100, 1)
        print(f"  {conf:<12} {len(cr):>5} {cwr:>5.1f}%")

    # ── Side breakdown ────────────────────────────────────────────────────────
    print(f"\n  {'SIDE':<8} {'BETS':>5} {'WIN%':>6}")
    print(f"  {'-'*22}")
    for side in ("YES", "NO"):
        sr  = [r for r in rows if r["best_side"] == side]
        if not sr: continue
        sw  = sum(1 for r in sr if r["won"] == 1)
        swr = round(sw / len(sr) * 100, 1)
        print(f"  {side:<8} {len(sr):>5} {swr:>5.1f}%")

    # ── Daily summary ─────────────────────────────────────────────────────────
    print(f"\n  {'DATE':<12} {'BETS':>5} {'WIN%':>6} {'CLV':>8}")
    print(f"  {'-'*35}")
    dates = sorted(set(r["pred_date"] for r in rows))
    for ds in dates:
        dr   = [r for r in rows if r["pred_date"] == ds]
        dw   = sum(1 for r in dr if r["won"] == 1)
        dwr  = round(dw / len(dr) * 100, 1)
        dclv = [r["clv"] for r in dr if r["clv"] is not None]
        davg = round(sum(dclv) / len(dclv), 1) if dclv else 0
        print(f"  {ds:<12} {len(dr):>5} {dwr:>5.1f}% {davg:>+7.1f}¢")

    print(f"\n{'='*55}\n")

# ── CLI ENTRY ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"

    if cmd == "score":
        # Optional: pass a date like `python bias_logger.py score 2026-05-24`
        if len(sys.argv) > 2:
            target = date.fromisoformat(sys.argv[2])
        else:
            target = None
        score(target)

    elif cmd == "report":
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        report(days)

    elif cmd == "init":
        init_db()
        print(f"[bias_logger] DB initialized at {DB_PATH}")

    elif cmd == "dump":
        # Quick raw dump of last 50 rows
        init_db()
        conn = _get_conn()
        rows = conn.execute(
            "SELECT pred_date, city_code, ticker, best_side, taker_ev, won, clv "
            "FROM predictions ORDER BY id DESC LIMIT 50"
        ).fetchall()
        conn.close()
        print(f"{'DATE':<12} {'CITY':<6} {'TICKER':<40} {'SIDE':<5} {'EV%':>6} {'WON':>4} {'CLV':>6}")
        print("-" * 85)
        for r in rows:
            won_str = "✅" if r["won"] == 1 else ("❌" if r["won"] == 0 else "—")
            clv_str = f"{r['clv']:+.0f}¢" if r["clv"] is not None else "—"
            print(f"{r['pred_date']:<12} {r['city_code']:<6} {r['ticker']:<40} "
                  f"{r['best_side']:<5} {r['taker_ev']:>+5.1f}% {won_str:>4} {clv_str:>6}")

    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python bias_logger.py [score|report|init|dump] [optional args]")
