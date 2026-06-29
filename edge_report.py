"""
edge_report.py — NO-book edge audit for the Kalshi weather bot.
Standalone + start-command friendly (no nested quotes, ASCII only).

    python3 edge_report.py            # last 60 days
    python3 edge_report.py --days 30

Accuracy does not pay -- PRICE does. A NO bought at 85c needs to win ~85% of
the time just to break even. Per entry-price band:
    breakeven% = avg entry cost | win% = actual win rate
    win% > breakeven%  => profitable band.
PAPER backtest of the alert signal, gross of Kalshi fees. Read-only.
"""

import os
import sys
from datetime import date, timedelta


def get_conn():
    import psycopg2
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        print("DATABASE_URL not set.")
        sys.exit(1)
    return psycopg2.connect(url)


def _fetch(days):
    cutoff = str(date.today() - timedelta(days=days))
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            SELECT no_price, model_correct, city_code, clv
            FROM predictions
            WHERE settled = 1
              AND best_side = 'NO'
              AND model_correct IS NOT NULL
              AND no_price IS NOT NULL
              AND LEFT(market_date, 10) >= %s
        """, (cutoff,))
        return cur.fetchall()
    finally:
        conn.close()


def _line(label, rows):
    n = len(rows)
    if n == 0:
        return None
    wins   = sum(r[1] for r in rows)
    cost   = sum(r[0] for r in rows)
    profit = sum(r[1] * 100 - r[0] for r in rows)
    be     = cost / n
    wp     = wins / n * 100
    roi    = profit / cost * 100 if cost else 0.0
    clvs   = [r[3] for r in rows if r[3] is not None]
    clv    = sum(clvs) / len(clvs) if clvs else None
    clv_s  = f"{clv:+5.1f}c" if clv is not None else "  n/a"
    flag   = "  <- PROFIT" if wp > be else ""
    return (f"{label:<10} n={n:>3} | breakeven {be:5.1f}% | win {wp:5.1f}% | "
            f"avgP {profit/n:+6.2f}c | ROI {roi:+6.1f}% | CLV {clv_s}{flag}")


def edge_report(days=60):
    rows = _fetch(days)
    print(f"NO-book edge -- last {days}d -- {len(rows)} settled bets")
    if not rows:
        print("(no settled NO bets in window)")
        return

    overall = _line("OVERALL", rows)
    if overall:
        print(overall)

    print("\n--- by entry-price band (win% > breakeven% = profitable) ---")
    bands = {}
    for r in rows:
        bands.setdefault((r[0] // 10) * 10, []).append(r)
    for b in sorted(bands):
        line = _line(f"{b:>2}-{b+9}c", bands[b])
        if line:
            print(line)

    print("\n--- by city (NO book, this window) ---")
    cities = {}
    for r in rows:
        cities.setdefault(r[2], []).append(r)
    ranked = []
    for cc, cr in cities.items():
        cost = sum(x[0] for x in cr)
        profit = sum(x[1] * 100 - x[0] for x in cr)
        roi = profit / cost * 100 if cost else 0.0
        ranked.append((roi, cc, cr))
    for _, cc, cr in sorted(ranked):
        line = _line(cc, cr)
        if line:
            print(line)


if __name__ == "__main__":
    args = sys.argv[1:]
    days = 60
    for a in args:
        if a.startswith("--days="):
            days = int(a.split("=")[1])
        elif a == "--days" and args.index(a) + 1 < len(args):
            days = int(args[args.index(a) + 1])
    edge_report(days=days)
