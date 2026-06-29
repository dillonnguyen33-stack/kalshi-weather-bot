"""
bias_audit.py — Per-city forecast bias audit (read-only, start-command friendly).

    python3 bias_audit.py             # all-time
    python3 bias_audit.py --days 60   # last 60 days only

For every SETTLED bet we compare the model's RAW prediction (ensemble_mean) to
the ACTUAL settled high (actual_high_f). Ignores the buggy bet_category field.

    raw_err  = avg(actual - ensemble_mean)   <- the bias the model NEEDS
               +value => model ran COLD ; -value => model ran HOT
    applied  = avg(bias_applied)             <- the bias the bot USED
    resid    = avg(actual - ensemble_mean - bias_applied)  <- leftover error
               near 0 = good ; large +/- = current bias is wrong (bleeding)
    suggest  = applied + damp(n) * (raw_err - applied)   <- corrected bias
               (FIXED math: blends toward the true target, not the residual)

Read-only. Does NOT write to the DB or change the bot.
"""

import os
import sys
from datetime import date, timedelta
from collections import defaultdict

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]

MIN_SAMPLES = 3   # below this we don't suggest a change


def get_conn():
    import psycopg2
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        print("DATABASE_URL not set.")
        sys.exit(1)
    return psycopg2.connect(url)


def damp(n):
    """Fraction of the gap (raw_err - applied) to apply, by sample size."""
    if n >= 15: return 0.85
    if n >= 8:  return 0.65
    if n >= 5:  return 0.55
    if n >= 3:  return 0.50
    return 0.0


def _fetch(days):
    conn = get_conn()
    try:
        cur = conn.cursor()
        where = ("WHERE settled = 1 AND actual_high_f IS NOT NULL "
                 "AND ensemble_mean IS NOT NULL")
        params = []
        if days:
            cutoff = str(date.today() - timedelta(days=days))
            where += " AND LEFT(market_date, 10) >= %s"
            params.append(cutoff)
        cur.execute(f"""
            SELECT city_code, market_date, actual_high_f, ensemble_mean,
                   COALESCE(bias_applied, 0)
            FROM predictions
            {where}
        """, params)
        return cur.fetchall()
    finally:
        conn.close()


def main():
    args = sys.argv[1:]
    days = 0
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--days="):
            days = int(a.split("=")[1])
        elif a == "--days" and i + 1 < len(args):
            days = int(args[i + 1]); i += 1
        i += 1

    rows = _fetch(days)
    window = "all-time" if not days else f"last {days}d"
    print(f"BIAS AUDIT ({window}) -- {len(rows)} settled rows with actual + ensemble_mean")
    if not rows:
        print("(no usable rows)")
        return

    # Aggregate by (city, month). Parse month from the date string directly to
    # avoid any SQL date-cast fragility on odd market_date values.
    agg = defaultdict(lambda: {"raw": [], "applied": [], "resid": []})
    skipped = 0
    for city, mdate, actual, ens, bias in rows:
        try:
            mon = int(str(mdate)[5:7])          # 'YYYY-MM-DD' -> MM
        except (ValueError, IndexError, TypeError):
            skipped += 1
            continue
        if not (1 <= mon <= 12):
            skipped += 1
            continue
        a = float(actual)
        if not (-50.0 <= a <= 140.0):           # sanity: real high temp range
            skipped += 1
            continue
        raw_err = a - float(ens)
        key = (city, mon - 1)
        agg[key]["raw"].append(raw_err)
        agg[key]["applied"].append(float(bias))
        agg[key]["resid"].append(raw_err - float(bias))

    if skipped:
        print(f"(skipped {skipped} rows with bad date or out-of-range temp)")

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    # Build per (city, month) summary
    summary = {}
    for (city, m0), d in agg.items():
        n = len(d["raw"])
        raw = avg(d["raw"])
        app = avg(d["applied"])
        res = avg(d["resid"])
        sug = app + damp(n) * (raw - app)
        summary[(city, m0)] = (n, raw, app, res, sug)

    cur_m0 = date.today().month - 1

    def line(city, m0, s):
        n, raw, app, res, sug = s
        flag = ""
        if n >= MIN_SAMPLES and abs(res) >= 1.0:
            flag = "  <- MIS-CORRECTED" if abs(res) >= 2.0 else "  <- off"
        return (f"  {city:<4} {MONTHS[m0]} n={n:>3} | raw_err {raw:+5.1f} | "
                f"applied {app:+5.1f} | resid {res:+5.1f} | suggest {sug:+5.2f}{flag}")

    # ── CURRENT MONTH, worst miscorrection first (the actionable view) ─────────
    print(f"\n=== CURRENT MONTH ({MONTHS[cur_m0]}) -- worst leftover error first ===")
    cur_keys = [(c, m0) for (c, m0) in summary if m0 == cur_m0]
    if not cur_keys:
        print("  (no settled rows in the current month yet)")
    else:
        cur_keys.sort(key=lambda k: -abs(summary[k][3]))  # by |resid|
        for (city, m0) in cur_keys:
            print(line(city, m0, summary[(city, m0)]))

    # ── FULL TABLE (every city x month with data) ─────────────────────────────
    print("\n=== ALL MONTHS (reference) ===")
    print("  raw_err: +=model COLD/needs +bias  |  resid: leftover after current bias")
    for (city, m0) in sorted(summary.keys()):
        print(line(city, m0, summary[(city, m0)]))

    print("\nNOTE: 'suggest' is the corrected bias (damped by sample size). Review")
    print("and we decide together which CITY_BIAS_F values to update in the bot.")


if __name__ == "__main__":
    main()
