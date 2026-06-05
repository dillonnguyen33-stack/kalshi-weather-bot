"""
One-time script to display June 3 results.
June 3 bets are stored as market_date = 2026-06-02 (old bug).
"""
import os, psycopg2, requests

DATABASE_URL        = os.environ.get("DATABASE_URL", "")
DISCORD_LOG_WEBHOOK = os.environ.get("DISCORD_LOG_WEBHOOK", "")

def get_conn():
    return psycopg2.connect(DATABASE_URL)

def run():
    conn = get_conn()
    cur  = conn.cursor()

    cur.execute("""
        SELECT ticker, best_side, yes_price, city_name, taker_ev,
               settled, yes_result, model_correct, clv
        FROM predictions
        WHERE market_date = '2026-06-02'
        ORDER BY city_name
    """)
    rows = cur.fetchall()
    conn.close()

    print(f"Found {len(rows)} June 3 predictions")

    wins = losses = pending = 0
    bet_lines = []

    for ticker, best_side, yes_price, city_name, taker_ev, settled, yes_result, model_correct, clv in rows:
        if not settled:
            pending += 1
            bet_lines.append(f"⏳ `{city_name}` {ticker.split('-')[-1]} — not settled")
            continue

        if model_correct:
            wins += 1
            icon = "✅"
        else:
            losses += 1
            icon = "❌"

        clv_str = f"CLV {clv:+.0f}¢" if clv is not None else ""
        result_str = "YES" if yes_result else "NO"
        bet_lines.append(f"{icon} `{city_name}` {ticker.split('-')[-1]} → {result_str} won | {clv_str}")
        print(bet_lines[-1])

    total = wins + losses
    acc   = f"{wins}/{total} = {wins/total*100:.1f}%" if total else "n/a"
    print(f"\nJune 3 Final: {acc} | {pending} pending")

    if DISCORD_LOG_WEBHOOK:
        embeds = [
            {
                "title": "📊 June 3 Scoreboard",
                "color": 0x57F287 if wins > losses else 0xED4245,
                "fields": [
                    {"name": "Wins",     "value": str(wins),    "inline": True},
                    {"name": "Losses",   "value": str(losses),  "inline": True},
                    {"name": "Pending",  "value": str(pending), "inline": True},
                    {"name": "Accuracy", "value": acc,          "inline": False},
                ]
            },
            {
                "title": "📋 Bet by Bet",
                "color": 0x5865F2,
                "description": "\n".join(bet_lines[:25]) if bet_lines else "No results",
            }
        ]
        requests.post(DISCORD_LOG_WEBHOOK,
                      json={"content": f"**June 3 Results — {acc}**", "embeds": embeds},
                      timeout=10)
        print("[discord] Posted to Discord")

if __name__ == "__main__":
    run()
