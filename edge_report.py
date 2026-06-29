"""
edge_report.py — NO-book edge audit with slippage haircut.
Standalone + start-command friendly (no nested quotes, ASCII only).

    python3 edge_report.py                 # last 60d, sensitivity sweep + detail at +0c
    python3 edge_report.py --slip 3        # detail breakdown priced at top-of-book +3c
    python3 edge_report.py --days 90 --slip 5

Stored no_price is the TOP-OF-BOOK ask. A real order walks the book to a worse
average fill. Modeled as a flat haircut: effective fill = no_price + slip (c),
capped at 99. Outcome unchanged -- you just pay more, so breakeven rises.
PAPER backtest, gross of fees. Read-only.
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
            SELECT no_price, model_correct, city_code
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


def _eff(no_price, slip):
    p = no_price + slip
    return p if p < 99 else 99


def _stats(rows, slip):
    n = len(rows)
    if n == 0:
        return None
    wins   = sum(r[1] for r in rows)
    cost   = sum(_eff(r[0], slip) for r in rows)
    profit = sum(r[1] * 100 - _eff(r[0], slip) for r in rows)
    be     = cost / n
    wp     = wins / n * 100
    roi    = profit / cost * 100 if cost else 0.0
    return n, be, wp, profit / n, roi


def _line(label, rows, slip):
    s = _stats(rows, slip)
    if not s:
        return None
    n, be, wp, avgp, roi = s
    flag = "  <- PROFIT" if wp > be else ""
    return (f"{label:<10} n={n:>3} | breakeven {be:5.1f}% | win {wp:5.1f}% | "
            f"avgP {avgp:+6.2f}c | ROI {roi:+6.1f}%{flag}")


def edge_report(days=60, slip=0):
    rows = _fetch(days)
    print(f"NO-book edge -- last {days}d -- {len(rows)} settled bets")
    if not rows:
        print("(no settled NO bets in window)")
        return

    print("\n--- slippage sensitivity (OVERALL NO book) ---")
    print("  effective fill = top-of-book + haircut; win rate fixed, you pay more")
    for s in (0, 2, 3, 5):
        st = _stats(rows, s)
        if st:
            n, be, wp, avgp, roi = st
            print(f"  +{s}c | breakeven {be:5.1f}% | win {wp:5.1f}% | "
                  f"avgP {avgp:+6.2f}c | ROI {roi:+6.1f}%")

    print(f"\n=== detail priced at top-of-book +{slip}c ===")

    print("\n--- by entry-price band (top-of-book band; win% > breakeven% = profit) ---")
    bands = {}
    for r in rows:
        bands.setdefault((r[0] // 10) * 10, []).append(r)
    for b in sorted(bands):
        line = _line(f"{b:>2}-{b+9}c", bands[b], slip)
        if line:
            print(line)

    print("\n--- by city (worst ROI first) ---")
    cities = {}
    for r in rows:
        cities.setdefault(r[2], []).append(r)
    ranked = []
    for cc, cr in cities.items():
        st = _stats(cr, slip)
        ranked.append((st[4], cc, cr))
    for _, cc, cr in sorted(ranked):
        line = _line(cc, cr, slip)
        if line:
            print(line)


if __name__ == "__main__":
    args = sys.argv[1:]
    days, slip = 60, 0
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--days="):
            days = int(a.split("=")[1])
        elif a == "--days" and i + 1 < len(args):
            days = int(args[i + 1]); i += 1
        elif a.startswith("--slip="):
            slip = int(a.split("=")[1])
        elif a == "--slip" and i + 1 < len(args):
            slip = int(args[i + 1]); i += 1
        i += 1
    edge_report(days=days, slip=slip)
