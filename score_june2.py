"""
One-time scorer for June 2 bets — fetches results directly from Kalshi
and posts scoreboard to Discord.
"""
import os, requests

DISCORD_LOG_WEBHOOK = os.environ.get("DISCORD_LOG_WEBHOOK", "")
KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"

BETS = [
    ("KXHIGHNY-26JUN02-B75.5",   "NO", "New York City",   75.5, "B"),
    ("KXHIGHLAX-26JUN02-B69.5",  "NO", "Los Angeles",     69.5, "B"),
    ("KXHIGHCHI-26JUN02-B74.5",  "NO", "Chicago",         74.5, "B"),
    ("KXHIGHMIA-26JUN02-T93",    "NO", "Miami",           93.0, "T"),
    ("KXHIGHMIA-26JUN02-B92.5",  "NO", "Miami",           92.5, "B"),
    ("KXHIGHTDC-26JUN02-B76.5",  "NO", "Washington DC",   76.5, "B"),
    ("KXHIGHTSEA-26JUN02-B85.5", "NO", "Seattle",         85.5, "B"),
    ("KXHIGHTPHX-26JUN02-B103.5","NO", "Phoenix",        103.5, "B"),
    ("KXHIGHTHOU-26JUN02-B89.5", "NO", "Houston",         89.5, "B"),
    ("KXHIGHTDAL-26JUN02-B96.5", "NO", "Dallas",          96.5, "B"),
    ("KXHIGHTLV-26JUN02-B100.5", "NO", "Las Vegas",      100.5, "B"),
    ("KXHIGHTBOS-26JUN02-B75.5", "NO", "Boston",          75.5, "B"),
    ("KXHIGHAUS-26JUN02-B92.5",  "NO", "Austin",          92.5, "B"),
    ("KXHIGHLAX-26JUN02-B71.5",  "NO", "Los Angeles",     71.5, "B"),
]

def fetch_result(ticker):
    try:
        r = requests.get(f"{KALSHI_BASE}/markets/{ticker}", timeout=10)
        r.raise_for_status()
        m = r.json().get("market", {})
        status = m.get("status", "")
        result = m.get("result", "")
        expiration_value = m.get("expiration_value", "")
        return status, result, expiration_value
    except Exception as e:
        return "error", "", str(e)

def score_bet(side, result):
    if not result: return None
    if side == "NO":
        return result == "no"
    return result == "yes"

def run():
    rows = []
    wins = losses = pending = 0

    for ticker, side, city, threshold, kind in BETS:
        status, result, exp_val = fetch_result(ticker)
        correct = score_bet(side, result)

        if correct is None:
            pending += 1
            icon = "⏳"
            outcome = "pending"
        elif correct:
            wins += 1
            icon = "✅"
            outcome = f"WON (result={result}, actual={exp_val}°)"
        else:
            losses += 1
            icon = "❌"
            outcome = f"LOST (result={result}, actual={exp_val}°)"

        line = f"{icon} `{city}` {ticker.split('-')[-1]} → {outcome}"
        rows.append(line)
        print(line)

    total = wins + losses + pending
    acc = f"{wins}/{wins+losses} = {wins/(wins+losses)*100:.1f}%" if (wins+losses) else "n/a"

    summary = f"**June 2 Results — NO Bets**\n✅ {wins} wins | ❌ {losses} losses | ⏳ {pending} pending\nAccuracy: {acc}"

    embeds = [
        {
            "title": "📊 June 2 Manual Scoreboard",
            "color": 0x57F287 if wins > losses else 0xED4245,
            "fields": [
                {"name": "Wins",    "value": str(wins),    "inline": True},
                {"name": "Losses",  "value": str(losses),  "inline": True},
                {"name": "Pending", "value": str(pending), "inline": True},
                {"name": "Accuracy","value": acc,          "inline": False},
            ]
        },
        {
            "title": "📋 Bet by Bet",
            "color": 0x5865F2,
            "description": "\n".join(rows),
        }
    ]

    if DISCORD_LOG_WEBHOOK:
        requests.post(DISCORD_LOG_WEBHOOK,
                      json={"content": summary, "embeds": embeds},
                      timeout=10)
        print("[discord] Scoreboard posted!")
    else:
        print("[discord] No webhook set — printing only")
        print(summary)

if __name__ == "__main__":
    run()
