"""
diag_report.py — Diagnose why calibration_report finds no settled bets.

Prints the raw market_date format, then runs a calibration with NO date
filter at all (just settled=1) so we can see the real numbers regardless
of how dates are stored.

Run: python3 diag_report.py
"""

import os
from collections import defaultdict

def get_conn():
    import psycopg2
    return psycopg2.connect(os.environ["DATABASE_URL"])

def main():
    conn = get_conn()
    cur = conn.cursor()

    # 1) Show raw market_date samples
    print("=" * 60)
    print("RAW market_date SAMPLES (settled bets)")
    print("=" * 60)
    cur.execute("SELECT market_date, settled, model_correct, best_side, bet_category "
                "FROM predictions WHERE settled=1 LIMIT 10")
    for r in cur.fetchall():
        print(f"  market_date={r[0]!r} settled={r[1]} correct={r[2]} side={r[3]} cat={r[4]!r}")

    # 2) Distinct market_date formats
    cur.execute("SELECT DISTINCT LENGTH(market_date) FROM predictions WHERE settled=1")
    print(f"\nDistinct market_date string lengths: {[r[0] for r in cur.fetchall()]}")

    # 3) Overall settled accuracy — NO date filter
    print("\n" + "=" * 60)
    print("CALIBRATION — ALL SETTLED BETS (no date filter)")
    print("=" * 60)
    cur.execute("""
        SELECT city_code, model_prob, best_side, model_correct,
               bet_category, spread, ensemble_mean
        FROM predictions WHERE settled = 1
    """)
    rows = cur.fetchall()
    total = len(rows)
    correct = sum(1 for r in rows if r[3] == 1)
    print(f"Total settled: {total}")
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.1f}%" if total else "none")

    # 4) Prob-bucket calibration
    print(f"\n{'Prob bucket':<14}{'Predicted':>10}{'Actual%':>9}{'N':>6}{'Bias':>8}")
    print("-" * 50)
    for lo, hi in [(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.01)]:
        bucket = [r for r in rows if r[1] is not None and lo <= r[1] < hi]
        if not bucket: continue
        n = len(bucket)
        wins = sum(1 for r in bucket if r[3] == 1)
        wp = wins/n*100
        mid = (lo+hi)/2*100
        print(f"{int(lo*100)}-{int(hi*100)}%{'':<8}{mid:>9.0f}%{wp:>8.1f}%{n:>6}{wp-mid:>+7.1f}%")

    # 5) By side
    print(f"\n{'Side':<10}{'N':>6}{'Acc%':>8}")
    print("-" * 26)
    side = defaultdict(lambda:[0,0])
    for r in rows:
        side[r[2]][0]+=1
        side[r[2]][1]+= (r[3] or 0)
    for s,(n,c) in side.items():
        print(f"{str(s):<10}{n:>6}{c/n*100:>7.1f}%")

    # 6) By category
    print(f"\n{'Category':<16}{'N':>6}{'Acc%':>8}")
    print("-" * 32)
    cat = defaultdict(lambda:[0,0])
    for r in rows:
        cat[r[4]][0]+=1
        cat[r[4]][1]+= (r[3] or 0)
    for k,(n,c) in sorted(cat.items(), key=lambda x:-x[1][0]):
        print(f"{str(k):<16}{n:>6}{c/n*100:>7.1f}%")

    # 7) By city (top 20 by volume)
    print(f"\n{'City':<10}{'N':>6}{'Acc%':>8}")
    print("-" * 26)
    city = defaultdict(lambda:[0,0])
    for r in rows:
        city[r[0]][0]+=1
        city[r[0]][1]+= (r[3] or 0)
    for k,(n,c) in sorted(city.items(), key=lambda x:-x[1][0])[:20]:
        print(f"{str(k):<10}{n:>6}{c/n*100:>7.1f}%")

    # 8) NO BETS ONLY — the current bot is NO-only (YES disabled ~a week ago)
    print("\n" + "=" * 60)
    print("NO BETS ONLY (reflects current bot)")
    print("=" * 60)
    no_rows = [r for r in rows if r[2] == "NO"]
    nt = len(no_rows)
    nc = sum(1 for r in no_rows if r[3] == 1)
    print(f"Total NO settled: {nt}")
    print(f"NO accuracy: {nc}/{nt} = {nc/nt*100:.1f}%" if nt else "none")

    # NO calibration buckets
    print(f"\n{'Prob bucket':<14}{'Pred':>7}{'Actual%':>9}{'N':>6}{'Bias':>8}")
    print("-" * 46)
    for lo, hi in [(0.50,0.60),(0.60,0.70),(0.70,0.80),(0.80,0.90),(0.90,1.01)]:
        b = [r for r in no_rows if r[1] is not None and lo <= r[1] < hi]
        if not b: continue
        n = len(b); wins = sum(1 for r in b if r[3]==1); wp = wins/n*100
        mid = (lo+hi)/2*100
        print(f"{int(lo*100)}-{int(hi*100)}%{'':<8}{mid:>6.0f}%{wp:>8.1f}%{n:>6}{wp-mid:>+7.1f}%")

    # NO bets by city
    print(f"\n{'City (NO)':<10}{'N':>6}{'Acc%':>8}")
    print("-" * 26)
    ncity = defaultdict(lambda:[0,0])
    for r in no_rows:
        ncity[r[0]][0]+=1
        ncity[r[0]][1]+= (r[3] or 0)
    for k,(n,c) in sorted(ncity.items(), key=lambda x:-x[1][0]):
        flag = "  <-- weak" if c/n < 0.55 else ""
        print(f"{str(k):<10}{n:>6}{c/n*100:>7.1f}%{flag}")

    conn.close()

if __name__ == "__main__":
    main()
