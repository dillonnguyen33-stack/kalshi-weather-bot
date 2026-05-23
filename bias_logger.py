"""
Kalshi Weather Bot — Bias Logger & Offset Calculator
=====================================================
Runs as a standalone daily job (cron or Task Scheduler) alongside v3.

What it does:
  Morning job (run ~9am local time, after Kalshi settles):
    1. Fetches all KXHIGH markets that settled yesterday
    2. Pulls the expiration_value (actual NWS temperature) from Kalshi API
    3. Fetches what your ensemble predicted for that city/date from the log
    4. Writes: date, city, threshold, result, actual_temp, predicted_temp, error
       into a local SQLite database (weather_bias.db)

  Bias report (run anytime):
    Reads the database and prints per-city, per-month mean error.
    Copy those numbers into CITY_BIAS_F in kalshi_weather_bot_v3.py.

Usage:
    python bias_logger.py log      # run morning settlement fetch
    python bias_logger.py report   # print current bias table
    python bias_logger.py report --min-days 30  # require 30+ samples per cell

SQLite schema:
    predictions  — what the bot predicted at scan time
    settlements  — what Kalshi settled at (actual NWS temp)
    bias_summary — materialized view updated on each report run

The predictions table is written by v3 when it scans a market.
Import `log_prediction` from this file into kalshi_weather_bot_v3.py.

Install: pip install requests  (sqlite3 is stdlib)
"""

import sqlite3, requests, json, sys, os
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

DB_PATH      = os.environ.get("BIAS_DB_PATH", "weather_bias.db")
KALSHI_BASE  = "https://trading-api.kalshi.com/trade-api/v2"
ET_TZ        = ZoneInfo("America/New_York")

# ── DATABASE SETUP ────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            logged_at       TEXT NOT NULL,          -- ISO timestamp when bot scanned
            market_date     TEXT NOT NULL,           -- YYYY-MM-DD settlement date
            city_code       TEXT NOT NULL,           -- e.g. NY, CH, LA
            city_name       TEXT NOT NULL,
            ticker          TEXT NOT NULL,           -- Kalshi market ticker
            threshold_f     REAL NOT NULL,           -- market threshold in °F
            ensemble_mean   REAL,                    -- raw ensemble mean °F
            corrected_mean  REAL,                    -- bias-corrected mean °F
            spread          REAL,                    -- ensemble spread °F
            ecmwf_high      REAL,
            hrrr_high       REAL,
            model_prob      REAL,                    -- probability model assigned
            yes_price       INTEGER,                 -- Kalshi yes price (cents)
            no_price        INTEGER,
            asos_obs_high   REAL,                    -- ASOS reading at scan time
            bias_applied    REAL,                    -- bias offset used
            UNIQUE(market_date, ticker, logged_at)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS settlements (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            market_date     TEXT NOT NULL,
            city_code       TEXT NOT NULL,
            ticker          TEXT NOT NULL UNIQUE,    -- one row per market
            threshold_f     REAL NOT NULL,
            result          TEXT,                    -- 'yes' or 'no'
            actual_temp_f   REAL,                    -- expiration_value from Kalshi
            settled_at      TEXT,                    -- ISO timestamp
            fetched_at      TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bias_summary (
            city_code       TEXT NOT NULL,
            month           INTEGER NOT NULL,        -- 1-12
            sample_count    INTEGER NOT NULL,
            mean_error_f    REAL NOT NULL,           -- actual - predicted (avg)
            std_error_f     REAL NOT NULL,
            last_updated    TEXT NOT NULL,
            PRIMARY KEY (city_code, month)
        )
    """)
    conn.commit()
    return conn


# ── PREDICTION LOGGER (called from v3 during scan) ───────────────────────────

def log_prediction(
    market_date: date,
    city_code: str,
    city_name: str,
    ticker: str,
    threshold_f: float,
    forecast: dict,
    model_prob: float,
    yes_price: int,
    no_price: int,
    asos_obs_high: float | None,
):
    """
    Call this from kalshi_weather_bot_v3.py inside scan_market_async
    after computing the forecast and probability. Records the prediction
    for later comparison against the actual settlement.

    Example call to add to v3 scan_market_async():
        from bias_logger import log_prediction
        log_prediction(
            today, cc, city_name, market["ticker"], threshold,
            forecast, prob, market["yes_price"], market["no_price"],
            obs_high,
        )
    """
    try:
        conn = get_db()
        conn.execute("""
            INSERT OR IGNORE INTO predictions
            (logged_at, market_date, city_code, city_name, ticker,
             threshold_f, ensemble_mean, corrected_mean, spread,
             ecmwf_high, hrrr_high, model_prob, yes_price, no_price,
             asos_obs_high, bias_applied)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            market_date.isoformat(),
            city_code, city_name, ticker,
            threshold_f,
            forecast.get("ensemble_mean"),
            forecast.get("corrected_mean"),
            forecast.get("spread"),
            forecast.get("ecmwf_high"),
            forecast.get("hrrr_high"),
            model_prob,
            yes_price, no_price,
            asos_obs_high,
            forecast.get("bias_applied", 0.0),
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[bias_logger] prediction log error: {e}")


# ── SETTLEMENT FETCHER ────────────────────────────────────────────────────────

def fetch_yesterday_settlements():
    """
    Fetches all KXHIGH markets that settled yesterday from Kalshi API.
    Kalshi's expiration_value field contains the actual NWS temperature.
    Writes results to the settlements table.
    """
    yesterday   = (date.today() - timedelta(days=1)).isoformat()
    yesterday_start = int(datetime.fromisoformat(f"{yesterday}T00:00:00+00:00").timestamp())
    yesterday_end   = int(datetime.fromisoformat(f"{yesterday}T23:59:59+00:00").timestamp())

    print(f"[settlement] Fetching settlements for {yesterday}...")

    markets, cursor = [], None
    while True:
        params = {
            "status": "settled",
            "min_close_ts": yesterday_start,
            "max_close_ts": yesterday_end,
            "limit": 100,
        }
        if cursor:
            params["cursor"] = cursor
        try:
            r = requests.get(f"{KALSHI_BASE}/markets", params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[settlement] API error: {e}"); break

        for m in data.get("markets", []):
            series = m.get("series_ticker", "")
            if not series.startswith("KXHIGH"):
                continue
            markets.append(m)

        cursor = data.get("cursor")
        if not cursor:
            break

    print(f"[settlement] Found {len(markets)} settled KXHIGH markets")
    conn = get_db()
    saved = 0

    for m in markets:
        ticker     = m.get("ticker", "")
        series     = m.get("series_ticker", "")
        city_code  = series.replace("KXHIGH", "")
        result     = m.get("result", "")           # 'yes' or 'no'
        exp_value  = m.get("expiration_value", "") # e.g. "72" or "72.0"
        subtitle   = m.get("subtitle", "")
        settled_at = m.get("latest_expiration_time", "")

        # Parse actual temp from expiration_value
        actual_temp = None
        if exp_value:
            try:
                actual_temp = float(exp_value)
            except ValueError:
                # Sometimes it's a range string like "72-74" — take midpoint
                import re
                nums = [float(n) for n in re.findall(r"[\d.]+", exp_value)]
                if nums:
                    actual_temp = sum(nums) / len(nums)

        # Parse threshold from subtitle
        import re as _re
        nums = [float(n) for n in _re.findall(r"[\d.]+", subtitle)]
        threshold = nums[0] if len(nums) == 1 else (sum(nums[:2]) / 2 if nums else None)

        if actual_temp is None:
            print(f"[settlement] No temp parsed for {ticker} (expiration_value='{exp_value}')")
            continue

        try:
            conn.execute("""
                INSERT OR REPLACE INTO settlements
                (market_date, city_code, ticker, threshold_f, result,
                 actual_temp_f, settled_at, fetched_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                yesterday, city_code, ticker, threshold,
                result, actual_temp, settled_at,
                datetime.now(timezone.utc).isoformat(),
            ))
            saved += 1
        except Exception as e:
            print(f"[settlement] DB error for {ticker}: {e}")

    conn.commit()
    conn.close()
    print(f"[settlement] Saved {saved} settlements to database")
    return saved


# ── BIAS CALCULATOR ───────────────────────────────────────────────────────────

def compute_bias_table(min_days: int = 20) -> dict:
    """
    Joins predictions and settlements to compute mean error per city per month.
    Error = actual_temp - corrected_mean_prediction.
    Returns dict matching CITY_BIAS_F format for copy-paste into v3.
    """
    conn = get_db()

    rows = conn.execute("""
        SELECT
            s.city_code,
            CAST(strftime('%m', s.market_date) AS INTEGER) AS month,
            s.actual_temp_f,
            p.corrected_mean,
            p.ensemble_mean,
            s.market_date
        FROM settlements s
        JOIN predictions p
          ON s.ticker = p.ticker
          AND s.market_date = p.market_date
        WHERE s.actual_temp_f IS NOT NULL
          AND p.corrected_mean IS NOT NULL
        ORDER BY s.city_code, month, s.market_date
    """).fetchall()

    conn.close()

    # Group by city + month
    from collections import defaultdict
    import statistics

    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in rows:
        error = row["actual_temp_f"] - row["corrected_mean"]
        grouped[(row["city_code"], row["month"])].append(error)

    # Build output table
    bias_table = {}
    all_cities = sorted(set(k[0] for k in grouped))

    for city in all_cities:
        monthly = []
        for month in range(1, 13):
            errors = grouped.get((city, month), [])
            if len(errors) >= min_days:
                mean_err = statistics.mean(errors)
                monthly.append(round(mean_err, 2))
            else:
                monthly.append(None)  # not enough data yet
        bias_table[city] = monthly

    return bias_table


def print_bias_report(min_days: int = 20):
    """Prints the computed bias table ready to paste into v3."""
    table = compute_bias_table(min_days)

    # Also compute raw stats
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM settlements").fetchone()[0]
    matched = conn.execute("""
        SELECT COUNT(*) FROM settlements s
        JOIN predictions p ON s.ticker=p.ticker AND s.market_date=p.market_date
        WHERE s.actual_temp_f IS NOT NULL AND p.corrected_mean IS NOT NULL
    """).fetchone()[0]
    conn.close()

    print(f"\n{'='*60}")
    print(f"  KALSHI WEATHER BIAS REPORT")
    print(f"  Total settlements in DB: {total}")
    print(f"  Matched to predictions:  {matched}")
    print(f"  Min samples required:    {min_days}/month for a cell to populate")
    print(f"{'='*60}\n")

    print("CITY_BIAS_F = {")
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    for city, monthly in sorted(table.items()):
        # Format: show value or 0.0 if not enough data (with comment)
        vals = []
        has_none = False
        for v in monthly:
            if v is None:
                vals.append("0.00")
                has_none = True
            else:
                vals.append(f"{v:+.2f}")
        row = ", ".join(vals)
        note = "  # INCOMPLETE — keep collecting" if has_none else ""
        print(f'    "{city}": [{row}],{note}')

    print("}")

    # Per-city summary
    print(f"\n{'='*60}")
    print("  CITY SUMMARY (mean error, std, sample count by month)")
    print(f"{'='*60}")
    print(f"  {'City':<6} {'Month':<6} {'Mean err':>9} {'Std':>7} {'N':>5}")
    print(f"  {'-'*40}")

    conn = get_db()
    rows = conn.execute("""
        SELECT
            s.city_code,
            CAST(strftime('%m', s.market_date) AS INTEGER) AS month,
            COUNT(*) as n,
            AVG(s.actual_temp_f - p.corrected_mean) as mean_err,
            AVG((s.actual_temp_f - p.corrected_mean) * (s.actual_temp_f - p.corrected_mean)) as mse
        FROM settlements s
        JOIN predictions p ON s.ticker=p.ticker AND s.market_date=p.market_date
        WHERE s.actual_temp_f IS NOT NULL AND p.corrected_mean IS NOT NULL
        GROUP BY s.city_code, month
        ORDER BY s.city_code, month
    """).fetchall()
    conn.close()

    for row in rows:
        std = row["mse"] ** 0.5 if row["mse"] else 0
        flag = " ✓" if row["n"] >= min_days else f" ({row['n']} samples)"
        print(f"  {row['city_code']:<6} {months[row['month']-1]:<6} "
              f"{row['mean_err']:>+8.2f}°F {std:>6.2f}°F {row['n']:>4}{flag}")

    print(f"\n  Values marked ✓ have {min_days}+ samples — reliable to use.")
    print(f"  Others need more data — leave as 0.00 for now.\n")


# ── DAILY AUTOMATION ──────────────────────────────────────────────────────────

def run_daily_log():
    """
    Full daily job:
    1. Fetch yesterday's settlements from Kalshi
    2. Recompute bias summary
    3. Print a brief status
    """
    saved = fetch_yesterday_settlements()

    conn = get_db()
    total_pred = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    total_set  = conn.execute("SELECT COUNT(*) FROM settlements").fetchone()[0]
    matched    = conn.execute("""
        SELECT COUNT(*) FROM settlements s
        JOIN predictions p ON s.ticker=p.ticker AND s.market_date=p.market_date
        WHERE s.actual_temp_f IS NOT NULL AND p.corrected_mean IS NOT NULL
    """).fetchone()[0]
    conn.close()

    print(f"\n[bias_logger] Daily log complete")
    print(f"  Predictions logged:   {total_pred}")
    print(f"  Settlements fetched:  {total_set}")
    print(f"  Matched pairs:        {matched}")
    print(f"  Run 'python bias_logger.py report' to see bias table")

    # Auto-print report once we have enough data
    if matched >= 100:
        print(f"\n[bias_logger] Enough data — printing bias table:")
        print_bias_report(min_days=20)


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "log"
    min_days = 20
    if "--min-days" in sys.argv:
        idx = sys.argv.index("--min-days")
        if idx + 1 < len(sys.argv):
            min_days = int(sys.argv[idx + 1])

    if cmd == "log":
        run_daily_log()
    elif cmd == "report":
        print_bias_report(min_days=min_days)
    elif cmd == "settlements":
        fetch_yesterday_settlements()
    else:
        print(f"Usage: python bias_logger.py [log|report|settlements] [--min-days N]")
        print(f"  log         — fetch yesterday's settlements + update DB")
        print(f"  report      — print current bias table (paste into v3)")
        print(f"  settlements — fetch settlements only, no report")
        sys.exit(1)
