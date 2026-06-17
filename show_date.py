"""
show_date.py — Show settled bet results for one specific date.

Usage: python3 show_date.py 2026-06-15
       (defaults to 2026-06-15 if no arg given)
"""

import os, sys
from collections import defaultdict

TARGET = sys.argv[1] if len(sys.argv) > 1 else "2026-06-15"

CATEGORY_EMOJI = {
    "overnight": "🌙", "morning": "🌅",
    "pacing": "📈", "pace_confirmed": "✅",
}

def get_conn():
    import psycopg2
    return psycopg2.connect(os.environ["DATABASE_URL"])

def main():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT city_code, ticker, threshold_f, threshold_kind, best_side,
               actual_high_f, model_correct, bet_category, model_prob,
               yes_price, no_price, taker_ev
        FROM predictions
        WHERE settled = 1
          AND LEFT(market_date, 10) = %s
        ORDER BY bet_category, city_code
    """, (TARGET,))
    rows = cur.fetchall()

    if not rows:
        print(f"No settled bets for {TARGET}")
        conn.close()
        return

    print("=" * 64)
    print(f"SETTLED BETS — {TARGET}")
    print("=" * 64)

    total = len(rows)
    correct = sum(1 for r in rows if r[6] == 1)
    no_n = sum(1 for r in rows if r[4] == "NO")
    no_c = sum(1 for r in rows if r[4] == "NO" and r[6] == 1)
    yes_n = sum(1 for r in rows if r[4] == "YES")
    yes_c = sum(1 for r in rows if r[4] == "YES" and r[6] == 1)

    print(f"Overall: {correct}/{total} = {correct/total*100:.1f}%")
    if no_n:  print(f"NO bets: {no_c}/{no_n} = {no_c/no_n*100:.1f}%")
    if yes_n: print(f"YES bets: {yes_c}/{yes_n} = {yes_c/yes_n*100:.1f}%")

    # Per-category
    print(f"\n{'Category':<16}{'N':>4}{'Acc%':>7}")
    print("-" * 28)
    cat = defaultdict(lambda: [0, 0])
    for r in rows:
        cat[r[7]][0] += 1
        cat[r[7]][1] += (r[6] or 0)
    for k, (n, c) in sorted(cat.items(), key=lambda x: -x[1][0]):
        emoji = CATEGORY_EMOJI.get(k, "")
        print(f"{emoji} {str(k):<13}{n:>4}{c/n*100:>6.0f}%")

    # Every bet
    print(f"\n{'Every bet':<40}")
    print("-" * 64)
    for (code, ticker, thresh, kind, side, actual, correct_b,
         category, mprob, yp, np_, ev) in rows:
        emoji = CATEGORY_EMOJI.get(category, "")
        res = "✅" if correct_b == 1 else "❌"
        actual_s = f"{actual:.0f}°F" if actual is not None else "?"

        # True bucket from ticker
        bucket_s = "?"
        if ticker and "-B" in ticker:
            try:
                b = float(ticker.rsplit("-B", 1)[1])
                bucket_s = f"{b-0.5:.0f}-{b+0.5:.0f}°"
            except: pass
        elif ticker and "-T" in ticker:
            try:
                t = float(ticker.rsplit("-T", 1)[1])
                bucket_s = (f"≥{t:.0f}°" if side == "YES" else f"<{t:.0f}°")
            except: pass

        ev_s = f"EV{ev:+.0f}%" if ev is not None else ""
        print(f"{res} {emoji} {code:<4} {side:<3} {bucket_s:<9} "
              f"→ {actual_s:<6} {ev_s}")

    conn.close()

if __name__ == "__main__":
    main()
