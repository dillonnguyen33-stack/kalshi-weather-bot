"""
morning_report.py — Daily calibration report posted to Discord

Posts every morning at 8am ET automatically.
Run via Railway cron: 0 13 * * * (13:00 UTC = 8:00 AM ET)

Usage:
  python3 morning_report.py        # run manually
"""

import os, requests
from datetime import date, timedelta
from zoneinfo import ZoneInfo

DATABASE_URL       = os.environ.get("DATABASE_URL", "")
CALIBRATION_WEBHOOK = os.environ.get(
    "CALIBRATION_WEBHOOK",
    "https://discordapp.com/api/webhooks/1509464133706190848/78HkHm6bKxzmRDZGSNy4YQNOAfJ2hPjh3fw6IRHolGAdrCQLYaimnqZDipj_kPGjV_NI"
)
ET_TZ = ZoneInfo("America/New_York")

def get_conn():
    import psycopg2
    return psycopg2.connect(DATABASE_URL)

def post_discord(content, embeds=None):
    try:
        requests.post(CALIBRATION_WEBHOOK,
                      json={"content": content, "embeds": embeds or []},
                      timeout=10)
    except Exception as e:
        print(f"[discord] {e}")

def run_report():
    conn = get_conn()
    cur  = conn.cursor()

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    cur.execute("SELECT COUNT(*) FROM predictions")
    total_logged = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM predictions WHERE settled = 1")
    total_settled = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM predictions WHERE settled = 0")
    total_pending = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM predictions WHERE model_correct = 1")
    total_correct = cur.fetchone()[0]

    # Yesterday's predictions
    yesterday = str(date.today() - timedelta(days=1))
    cur.execute("SELECT COUNT(*) FROM predictions WHERE market_date = %s", (yesterday,))
    yesterday_logged = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE market_date = %s AND settled = 1
    """, (yesterday,))
    yesterday_settled = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM predictions
        WHERE market_date = %s AND model_correct = 1
    """, (yesterday,))
    yesterday_correct = cur.fetchone()[0]

    # ── PROBABILITY BUCKETS ───────────────────────────────────────────────────
    cur.execute("""
        SELECT model_prob, model_correct
        FROM predictions
        WHERE settled = 1
    """)
    settled_rows = cur.fetchall()

    bucket_lines = []
    for lo, hi in [(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.00)]:
        bucket = [(p, c) for p, c in settled_rows if p is not None and lo <= p < hi]
        if not bucket:
            continue
        n    = len(bucket)
        wins = sum(1 for _, c in bucket if c == 1)
        win_pct = wins / n * 100
        mid     = (lo + hi) / 2 * 100
        bias    = win_pct - mid
        flag = " ⚠️ overconfident" if bias < -5 else (" ✅ underconfident" if bias > 5 else "")
        bucket_lines.append(
            f"`{int(lo*100)}-{int(hi*100)}%` → {win_pct:.0f}% actual  (n={n}, bias={bias:+.1f}%){flag}"
        )

    # ── CITY BREAKDOWN ────────────────────────────────────────────────────────
    cur.execute("""
        SELECT city_code, COUNT(*) as n,
               SUM(CASE WHEN model_correct = 1 THEN 1 ELSE 0 END) as correct,
               AVG(taker_ev) as avg_ev,
               AVG(model_prob) as avg_prob
        FROM predictions
        WHERE settled = 1
        GROUP BY city_code
        ORDER BY n DESC
    """)
    city_rows = cur.fetchall()

    city_lines = []
    for code, n, correct, avg_ev, avg_prob in city_rows:
        acc  = (correct / n * 100) if n else 0
        ev_s = f"{avg_ev:.1f}%" if avg_ev else "n/a"
        gap  = (correct / n) - avg_prob if avg_prob else 0
        flag = " ⚠️" if gap < -0.08 else ""
        city_lines.append(f"`{code:<6}` {n:>3} bets  {acc:>5.1f}% acc  EV {ev_s}{flag}")

    conn.close()

    # ── BUILD DISCORD MESSAGE ─────────────────────────────────────────────────
    today_str = date.today().strftime("%B %d, %Y")
    acc_str   = f"{total_correct}/{total_settled} = {total_correct/total_settled*100:.1f}%" if total_settled else "n/a"
    yest_acc  = f"{yesterday_correct}/{yesterday_settled} = {yesterday_correct/yesterday_settled*100:.1f}%" if yesterday_settled else "n/a"

    embeds = []

    # Summary embed
    embeds.append({
        "title": f"📊 Daily Calibration Report — {today_str}",
        "color": 0x5865F2,
        "fields": [
            {"name": "Total Logged",   "value": str(total_logged),   "inline": True},
            {"name": "Total Settled",  "value": str(total_settled),  "inline": True},
            {"name": "Pending",        "value": str(total_pending),  "inline": True},
            {"name": "Overall Accuracy", "value": acc_str,           "inline": True},
            {"name": f"Yesterday ({yesterday})",
             "value": f"{yesterday_logged} logged / {yesterday_settled} settled / {yest_acc} acc",
             "inline": False},
        ]
    })

    # Probability bucket embed
    if bucket_lines:
        embeds.append({
            "title": "🎯 Model Calibration by Probability Bucket",
            "color": 0x57F287,
            "description": "\n".join(bucket_lines),
            "footer": {"text": "Overconfident = winning less than predicted | Underconfident = winning more"}
        })
    else:
        embeds.append({
            "title": "🎯 Model Calibration",
            "color": 0x57F287,
            "description": "Not enough settled predictions yet. Keep logging — data builds up fast.",
        })

    # City breakdown embed
    if city_lines:
        embeds.append({
            "title": "🏙️ Accuracy by City",
            "color": 0xFEE75C,
            "description": "\n".join(city_lines[:15]),  # cap at 15 cities
            "footer": {"text": "⚠️ = model prob significantly higher than actual win rate — recalibrate"}
        })

    post_discord("", embeds=embeds)
    print(f"[morning_report] Posted to Discord — {total_settled} settled, {acc_str} accuracy")

if __name__ == "__main__":
    run_report()
