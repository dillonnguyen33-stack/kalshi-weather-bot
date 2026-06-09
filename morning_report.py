"""
morning_report.py — Daily calibration report posted to Discord

Posts every afternoon at 3pm PT automatically.
Run via Railway cron: 0 23 * * * (23:00 UTC = 3:00 PM PT)

v3.27: Detailed scoreboard — lists each settled bet with the final
       temperature, the bucket/threshold, win/loss, and bet category
       (overnight/morning/pacing). Adds a category accuracy breakdown.

Usage:
  python3 morning_report.py        # run manually
"""

import os, requests
from datetime import date, timedelta
from zoneinfo import ZoneInfo

DATABASE_URL        = os.environ.get("DATABASE_URL", "")
DISCORD_LOG_WEBHOOK = os.environ.get("DISCORD_LOG_WEBHOOK", "")
ET_TZ = ZoneInfo("America/New_York")

CATEGORY_EMOJI = {
    "overnight": "🌙",
    "morning":   "🌅",
    "pacing":    "📈",
}

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

    # ── DETAILED PER-BET BREAKDOWN ────────────────────────────────────────────
    cur.execute("""
        SELECT city_code, ticker, threshold_f, threshold_kind, best_side,
               actual_high_f, model_correct, bet_category, ensemble_mean
        FROM predictions
        WHERE market_date = %s AND settled = 1
        ORDER BY bet_category, city_code
    """, (str(yesterday),))
    bet_rows = cur.fetchall()

    bet_lines = []
    for (code, ticker, thresh, kind, side, actual, correct_b,
         category, mean) in bet_rows:
        emoji  = CATEGORY_EMOJI.get(category, "")
        result = "✅" if correct_b == 1 else "❌"
        actual_s = f"{actual:.0f}°F" if actual is not None else "?"

        # Describe the bucket/threshold
        if kind == "B":
            lo = thresh - 0.5
            hi = thresh + 0.5
            bucket_s = f"{lo:.0f}-{hi:.0f}°"
        else:
            bucket_s = f"≥{thresh:.0f}°" if side == "YES" else f"<{thresh:.0f}°"

        bet_lines.append(
            f"{result} {emoji} `{code:<4}` {side} {bucket_s} → settled {actual_s}"
        )

    # ── CATEGORY BREAKDOWN ────────────────────────────────────────────────────
    cur.execute("""
        SELECT bet_category, COUNT(*),
               SUM(CASE WHEN model_correct=1 THEN 1 ELSE 0 END)
        FROM predictions
        WHERE market_date = %s AND settled = 1
        GROUP BY bet_category
    """, (str(yesterday),))
    cat_rows = cur.fetchall()

    cat_lines = []
    for cat, n, c in cat_rows:
        emoji = CATEGORY_EMOJI.get(cat, "")
        acc = (c or 0) / n * 100 if n else 0
        cat_lines.append(f"{emoji} `{(cat or 'none'):<10}` {c}/{n} = {acc:.0f}%")

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

    if cat_lines:
        embeds.append({
            "title": "🕐 By Category",
            "color": 0xEB459E,
            "description": "\n".join(cat_lines),
        })

    if bet_lines:
        # Discord embed description limit ~4096 chars — chunk if needed
        chunk = []
        char_count = 0
        for line in bet_lines:
            if char_count + len(line) > 3900:
                embeds.append({
                    "title": "🧾 Every Bet (settled)",
                    "color": 0x57F287,
                    "description": "\n".join(chunk),
                })
                chunk = []
                char_count = 0
            chunk.append(line)
            char_count += len(line) + 1
        if chunk:
            embeds.append({
                "title": "🧾 Every Bet (settled)" if not any(
                    e.get("title") == "🧾 Every Bet (settled)" for e in embeds
                ) else "🧾 Every Bet (cont.)",
                "color": 0x57F287,
                "description": "\n".join(chunk),
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

    # Category breakdown for the week
    cur.execute("""
        SELECT bet_category, COUNT(*),
               SUM(CASE WHEN model_correct = 1 THEN 1 ELSE 0 END)
        FROM predictions
        WHERE market_date >= %s AND market_date < %s AND settled = 1
        GROUP BY bet_category
    """, (str(week_start), str(today)))
    cat_rows = cur.fetchall()

    cat_lines = []
    for cat, n, c in cat_rows:
        emoji = CATEGORY_EMOJI.get(cat, "")
        acc = (c or 0) / n * 100 if n else 0
        cat_lines.append(f"{emoji} `{(cat or 'none'):<10}` {c}/{n} = {acc:.0f}%")

    embeds = []
    embeds.append({
        "title": f"📈 Weekly Summary — {week_start} to {today - timedelta(days=1)}",
        "color": 0x5865F2,
        "fields": [
            {"name": "Week NO Accuracy", "value": week_acc, "inline": True},
            {"name": "Total NO Settled", "value": str(total_no_settled), "inline": True},
        ]
    })

    if cat_lines:
        embeds.append({
            "title": "🕐 By Category (Week)",
            "color": 0xEB459E,
            "description": "\n".join(cat_lines),
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
