"""
One-time script to score and display June 3 results.
June 3 bets are stored as market_date = 2026-06-02 (old bug).
"""
import os, psycopg2, requests

DATABASE_URL        = os.environ.get("DATABASE_URL", "")
DISCORD_LOG_WEBHOOK = os.environ.get("DISCORD_LOG_WEBHOOK", "")
KALSHI_BASE         = "https://external-api.kalshi.com/trade-api/v2"

def get_conn():
    return psycopg2.connect(DATABASE_URL)

def fetch_result(ticker):
    try:
        r = requests.get(f"{KALSHI_BASE}/markets/{ticker}", timeout=10)
        r.raise_for_status()
        m = r.json().get("market", {})
        if m.get("status") not in ("settled", "finalized"):
            return None
        result = m.get("result", "")
        if not result:
            return None
        yes_result = 1 if result == "yes" else 0
        exp_val = m.get("expiration_value", "")
        closing = m.get("close_price") or m.get("yes_bid") or m.get("last_price")
        closing_cents = round(float(closing) * 100) if closing else None
        return {"yes_result": yes_result, "closing_yes_price": closing_cents, "actual": exp_val}
    except Exception as e:
        print(f"[error] {ticker}: {e}")
        return None

def run():
    conn = get_conn()
    cur  = conn.cursor()

    # June 3 bets stored under market_date 2026-06-02 due to old bug
    cur.execute("""
        SELECT id, ticker, best_side, yes_price, model_prob, city_name, taker_ev
        FROM predictions
        WHERE market_date = '2026-06-02' AND settled = 0
        ORDER BY city_name
    """)
    rows = cur.fetchall()
    print(f"Found {len(rows)} unsettled June 3 predictions")

    wins = losses = pending = 0
    bet_lines = []

    for rid, ticker, best_side, yes_price, model_prob, city_name, taker_ev in rows:
        result = fetch_result(ticker)
        if result is None:
            pending += 1
            bet_lines.append(f"⏳ `{city_name}` {ticker.split('-')[-1]} — not settled")
            continue

        yes_result    = result["yes_result"]
        closing_price = result["closing_yes_price"]
        actual        = result["actual"]

        if best_side == "YES":
            model_correct = 1 if yes_result == 1 else 0
            entry_price   = yes_price
        else:
            model_correct = 1 if yes_result == 0 else 0
            entry_price   = 100 - yes_price

        clv = None
        if closing_price is not None:
            if best_side == "YES":
                clv = round(closing_price - entry_price, 1)
            else:
                clv = round((100 - closing_price) - entry_price, 1)

        # Update DB
        cur.execute("""
            UPDATE predictions
            SET settled=1, yes_result=%s, model_correct=%s,
                closing_yes_price=%s, clv=%s
            WHERE id=%s
        """, (yes_result, model_correct, closing_price, clv, rid))

        if model_correct:
            wins += 1
            icon = "✅"
        else:
            losses += 1
            icon = "❌"

        clv_str = f"CLV {clv:+.0f}¢" if clv is not None else ""
        bet_lines.append(f"{icon} `{city_name}` {ticker.split('-')[-1]} → {'YES' if yes_result else 'NO'} won | actual={actual}° | {clv_str}")
        print(bet_lines[-1])

    conn.commit()
    conn.close()

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
                "description": "\n".join(bet_lines[:25]),
            }
        ]
        requests.post(DISCORD_LOG_WEBHOOK,
                      json={"content": f"**June 3 Results — {acc}**", "embeds": embeds},
                      timeout=10)
        print("[discord] Posted to Discord")

if __name__ == "__main__":
    run()
