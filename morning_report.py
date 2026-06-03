"""
morning_report.py — Daily calibration report posted to Discord

Posts every afternoon at 3pm PT automatically.
Run via Railway cron: 0 23 * * * (23:00 UTC = 3:00 PM PT)

Usage:
  python3 morning_report.py        # run manually
"""

import os, requests
from datetime import date, timedelta
from zoneinfo import ZoneInfo

DATABASE_URL        = os.environ.get("DATABASE_URL", "")
DISCORD_LOG_WEBHOOK = os.environ.get("DISCORD_LOG_WEBHOOK", "")
ET_TZ = ZoneInfo("America/New_York")

def get_conn():
    import psycopg2
    return psycopg2.connect(DATABASE_URL)

def post_discord(content, embeds=None):
    webhook = DISCORD_LOG_WEBHOOK
    if not webhook:
        print("[discord] No webhook set")
        return
    try:
        requests.post(webhook,
                      json={"content": content, "embeds": embeds or []},
                      timeout=10)
    except Exception as e:
        print(f"[discord] {e}")

def run_daily_report(cur, yesterday):
    """Daily report — yesterday's results only."""

    # ── YESTERDAY SUMMARY ─────────────────────────────────────────────────────
    cur.execute("""
        SELECT COUNT(*),
               SUM(CASE WHEN settled = 1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN model_correct = 1 THEN 1 ELSE 0 END),
               SUM(CASE WHEN settled = 1 AND best_side = 'NO' THEN 1 ELSE 0 END),
               SUM(CASE WHEN model_correct = 1 AND best_side = 'NO' THEN 1 ELSE 0 END),
               SUM(CASE WHEN settled = 1 AND best_side = 'YES' THEN 1 ELSE 0 END),
               SUM(CASE WHEN model_correct = 1 AND best_side = 'YES' THEN 1 ELSE 0 END)
        FROM predictions
        WHERE market_date = %s
    """, (str(yesterday),))
    row = cur.fetchone()
    total, settled, correct, no_settled, no_correct, yes_settled, yes_correct = row

    pending = (total or 0) - (settled or 0)

    no_acc_str  = f"{no_correct}/{no_settled} = {no_correct/no_settled*100:.1f}%" if no_settled else "n/a"
    yes_acc_str = f"{yes_correct}/{yes_settled} = {yes_correct/yes_settled*100:.1f}%" if yes_settled else "n/a"
    all_acc_str = f"{correct}/{settled} = {correct/settled*100:.1f}%" if settled else "n/a"

    # ── YESTERDAY B-TYPE vs T-TYPE (NO bets only) ─────────────────────────────
    cur.execute("""
        SELECT threshold_kind, COUNT(*),
               SUM(CASE WHEN model_correct = 1 THEN 1 ELSE 0 END)
        FROM predictions
        WHERE market_date = %s AND settled = 1 AND best_side = 'NO'
        GROUP BY threshold_kind
    """, (str(yesterday),))
    type_rows = cur.fetchall()

    type_lines = []
    for kind, n, correct_t in type_rows:
        acc = correct_t / n * 100 if n else 0
        label = "Bucket (B)" if kind == "B" else "Threshold (T)"
        type_lines.append(f"`{label}` {n} bets → {correct_t}/{n} = {acc:.0f}% win rate")

    # ── YESTERDAY CITY BREAKDOWN (NO bets only) ───────────────────────────────
    cur.execute("""
        SELECT city_code, COUNT(*),
               SUM(CASE WHEN model_correct = 1 THEN 1 ELSE 0 END),
               AVG(taker_ev)
        FROM predictions
        WHERE market_date = %s AND settled = 1 AND best_side = 'NO'
        GROUP BY city_code
        ORDER BY COUNT(*) DESC
    """, (str(yesterday),))
    city_rows = cur.fetchall()

    city_lines = []
    for code, n, correct_c, avg_ev in city_rows:
        acc  = correct_c / n * 100 if n else 0
        ev_s = f"{avg_ev:.1f}%" if avg_ev else "n/a"
        city_lines.append(f"`{code:<6}` {n:>2} bets  {acc:>5.1f}% acc  EV {ev_s}")

    # ── BUILD EMBEDS ──────────────────────────────────────────────────────────
    embeds = []

    embeds.append({
        "title": f"📊 Daily Report — {yesterday}",
        "color": 0x5865F2,
        "fields": [
            {"name": "Logged",    "value": str(total or 0),   "inline": True},
            {"name": "Settled",   "value": str(settled or 0), "inline": True},
            {"name": "Pending",   "value": str(pending),      "inline": True},
            {"name": "NO Bets",   "value": no_acc_str,        "inline": True},
            {"name": "YES Bets",  "value": yes_acc_str,       "inline": True},
            {"name": "Overall",   "value": all_acc_str,       "inline": True},
        ]
    })

    if type_lines:
        embeds.append({
            "title": "📊 B-Type vs T-Type (NO bets)",
            "color": 0x57F287,
            "description": "\n".join(type_lines),
        })

    if city_lines:
        embeds.append({
            "title": "🏙️ City Breakdown (NO bets)",
            "color": 0xFEE75C,
            "description": "\n".join(city_lines[:15]),
        })

    if not settled:
        embeds.append({
            "title": "⏳ No settled bets yet",
            "color": 0xED4245,
            "description": f"Markets from {yesterday} haven't settled yet. Check back later.",
        })

    return embeds

def run_weekly_report(cur, today):
    """Weekly summary — fires on Mondays only."""

    week_start = today - timedelta(days=7)

    cur.execute("""
        SELECT market_date,
               SUM(CASE WHEN settled = 1 AND best_side = 'NO' THEN 1 ELSE 0 END) as no_settled,
               SUM(CASE WHEN model_correct = 1 AND best_side = 'NO' THEN 1 ELSE 0 END) as no_correct
        FROM predictions
        WHERE market_date >= %s AND market_date < %s
        GROUP BY market_date
        ORDER BY market_date
    """, (str(week_start), str(today)))
    weekly_rows = cur.fetchall()

    if not weekly_rows:
        return None

    total_no_settled = sum(r[1] for r in weekly_rows)
    total_no_correct = sum(r[2] for r in weekly_rows)
    week_acc = f"{total_no_correct}/{total_no_settled} = {total_no_correct/total_no_settled*100:.1f}%" if total_no_settled else "n/a"

    day_lines = []
    for mdate, no_settled, no_correct in weekly_rows:
        acc = f"{no_correct}/{no_settled} = {no_correct/no_settled*100:.0f}%" if no_settled else "no data"
        day_lines.append(f"`{mdate}` NO bets: {acc}")

    # City breakdown for the week
    cur.execute("""
        SELECT city_code, COUNT(*),
               SUM(CASE WHEN model_correct = 1 THEN 1 ELSE 0 END),
               AVG(taker_ev)
        FROM predictions
        WHERE market_date >= %s AND market_date < %s
              AND settled = 1 AND best_side = 'NO'
        GROUP BY city_code
        ORDER BY COUNT(*) DESC
    """, (str(week_start), str(today)))
    city_rows = cur.fetchall()

    city_lines = []
    for code, n, correct, avg_ev in city_rows:
        acc  = correct / n * 100 if n else 0
        ev_s = f"{avg_ev:.1f}%" if avg_ev else "n/a"
        city_lines.append(f"`{code:<6}` {n:>3} bets  {acc:>5.1f}% acc  EV {ev_s}")

    embeds = []
    embeds.append({
        "title": f"📈 Weekly Summary — {week_start} to {today - timedelta(days=1)}",
        "color": 0x5865F2,
        "fields": [
            {"name": "Week NO Accuracy", "value": week_acc, "inline": True},
            {"name": "Total NO Settled", "value": str(total_no_settled), "inline": True},
        ]
    })

    if day_lines:
        embeds.append({
            "title": "📅 Day by Day",
            "color": 0x57F287,
            "description": "\n".join(day_lines),
        })

    if city_lines:
        embeds.append({
            "title": "🏙️ City Breakdown (Week)",
            "color": 0xFEE75C,
            "description": "\n".join(city_lines[:15]),
        })

    return embeds

def run_report():
    conn = get_conn()
    cur  = conn.cursor()
    today     = date.today()
    yesterday = today - timedelta(days=1)

    # Daily report — always runs
    daily_embeds = run_daily_report(cur, yesterday)
    post_discord("", embeds=daily_embeds)
    print(f"[morning_report] Daily report posted for {yesterday}")

    # Weekly report — only on Mondays (weekday 0)
    if today.weekday() == 0:
        weekly_embeds = run_weekly_report(cur, today)
        if weekly_embeds:
            post_discord("", embeds=weekly_embeds)
            print(f"[morning_report] Weekly report posted")

    conn.close()

if __name__ == "__main__":
    run_report()
