"""
derive_bias.py — Derive real per-city per-month bias from logged outcomes

WHAT THIS DOES
  The bot's CITY_BIAS_F table is hand-estimated. This replaces it with bias
  computed from data the bot has actually logged.

  Bias is defined as:  error = settled_high - corrected_mean
  A POSITIVE mean error means the model ran COLD (actual was hotter than the
  model predicted) -> the bias correction should be POSITIVE (add degrees).
  A NEGATIVE mean error means the model ran HOT -> bias should be NEGATIVE.

  This matches how get_bias() is used in the bot:
      corrected_mean = mean + bias
  so bias[city][month] should equal the average (settled - mean) so that the
  correction cancels the historical error.

TWO DATA SOURCES
  1. predictions  (PRIMARY) — the bot's own forecasts vs settled outcomes.
     This measures the ACTUAL bias of the current blend. This is the fix.
  2. historical_highs (REFERENCE ONLY) — 5 years of actual daily highs.
     Cannot measure model bias by itself (no stored forecast to compare to),
     but is used here to report climatology and to warn when a city has too
     few logged predictions to trust.

OUTPUT
  - Prints a per-city / per-month table of measured bias + sample counts.
  - Prints a ready-to-paste CITY_BIAS_F dict. Cities with too few samples in
    a given month keep their EXISTING hand-coded value (we do not overwrite
    on thin data).

RUN
  Set DATABASE_URL, then: python3 derive_bias.py
  Read-only. It does NOT write to the database or modify the bot.
"""

import os
import sys
from collections import defaultdict

try:
    import psycopg2
except ImportError:
    print("psycopg2 not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

# ── Minimum logged samples required before we trust a derived value ──────────
MIN_SAMPLES_PER_MONTH = 5      # below this, keep the existing hand value
STRONG_SAMPLES        = 15     # at/above this, treat the derived value as solid

# ── EXISTING hand-coded table (so thin-data months are preserved verbatim) ───
# Paste from the bot. Index 0 = January ... index 11 = December.
EXISTING_CITY_BIAS_F = {
    "NY":  [ 0.8,  0.7,  0.5,  0.3,  0.2,  0.0, -0.3, -0.2,  0.0,  0.3,  0.5,  0.7],
    "CHI": [ 1.2,  1.0,  0.8,  0.4,  0.2,  0.0, -0.5, -0.4,  0.0,  0.5,  0.8,  1.1],
    "LAX": [-0.5, -0.4, -0.3, -0.2, -0.2, -0.3, -0.4, -0.4, -0.3, -0.2, -0.3, -0.4],
    "MIA": [ 0.3,  0.3,  0.2,  0.1,  0.0,  2.5,  2.5,  2.5,  0.0,  0.1,  0.2,  0.3],
    "PH":  [ 0.7,  0.6,  0.5,  0.3,  0.1,  0.0, -0.3, -0.2,  0.0,  0.3,  0.5,  0.6],
    "AT":  [ 0.5,  0.4,  0.3,  0.2,  0.1,  0.0, -0.3, -0.3, -0.1,  0.2,  0.3,  0.4],
    "MN":  [ 1.5,  1.3,  1.0,  0.5,  0.2,  0.0, -0.5, -0.4,  0.0,  0.6,  1.0,  1.4],
    "SF":  [-0.8, -0.7, -0.6, -0.5, -0.5, -0.6, -0.7, -0.7, -0.6, -0.5, -0.6, -0.7],
    "DA":  [ 0.3,  0.2,  0.1,  0.0, -0.1, -0.3, -0.6, -0.5, -0.2,  0.0,  0.2,  0.3],
    "BO":  [ 0.9,  0.8,  0.6,  0.4,  0.2,  0.0, -0.3, -0.2,  0.0,  0.4,  0.6,  0.8],
    "PHX": [-0.4, -0.3, -0.2, -0.1,  0.0, -0.3, -0.8, -0.7, -0.3, -0.1, -0.2, -0.3],
    "DEN": [ 0.6,  0.5,  0.4,  0.2,  0.1,  0.0, -0.4, -0.3,  0.0,  0.3,  0.5,  0.6],
    "SE":  [-0.3, -0.3, -0.2, -0.1,  0.0,  0.0, -0.2, -0.2, -0.1,  0.0, -0.2, -0.3],
    "HO":  [ 0.2,  0.1,  0.0, -0.1, -0.2, -0.4, -0.6, -0.6, -0.3, -0.1,  0.1,  0.2],
    "LV":  [-0.5, -0.4, -0.2,  0.0,  0.1, -0.2, -0.8, -0.7, -0.2,  0.0, -0.2, -0.4],
}
DEFAULT_BIAS = [0.0] * 12

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]


def connect():
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        print("DATABASE_URL not set.")
        sys.exit(1)
    return psycopg2.connect(url)


def describe_table(cur, table):
    """Return list of (column, type) for a table, or None if it doesn't exist."""
    cur.execute("""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_name = %s
        ORDER BY ordinal_position
    """, (table,))
    rows = cur.fetchall()
    return rows or None


def pick_column(columns, *candidates):
    """Find the first candidate column name that actually exists (case-insensitive)."""
    lower = {c[0].lower(): c[0] for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def derive_from_predictions(cur):
    """
    Returns: bias[city][month_index_0_11] = (mean_error, n_samples)
    where mean_error = mean(settled - corrected_mean).
    """
    cols = describe_table(cur, "predictions")
    if cols is None:
        print("[predictions] table not found — cannot derive live bias.")
        return {}, None

    print("\n[predictions] columns found:")
    for name, typ in cols:
        print(f"    {name:24s} {typ}")

    # Resolve the columns we need, tolerating naming variations.
    city_col    = pick_column(cols, "city_code", "city", "code")
    settled_col = pick_column(cols, "settled_high", "actual_high", "settled",
                              "final_high", "obs_high")
    mean_col    = pick_column(cols, "corrected_mean", "ensemble_mean",
                              "model_mean", "mean")
    date_col    = pick_column(cols, "market_date", "target_date", "date",
                              "logged_at", "created_at")

    missing = [n for n, c in [("city", city_col), ("settled", settled_col),
                              ("corrected_mean", mean_col), ("date", date_col)]
               if c is None]
    if missing:
        print(f"\n[predictions] Missing required column(s): {missing}")
        print("Edit pick_column() candidates to match your schema, then rerun.")
        return {}, None

    print(f"\n[predictions] using: city={city_col} settled={settled_col} "
          f"mean={mean_col} date={date_col}")

    # Only rows where the bet actually settled (settled high is present).
    cur.execute(f"""
        SELECT {city_col}, {mean_col}, {settled_col},
               EXTRACT(MONTH FROM {date_col}::timestamp)::int AS mon
        FROM predictions
        WHERE {settled_col} IS NOT NULL
          AND {mean_col}    IS NOT NULL
    """)
    rows = cur.fetchall()

    errors = defaultdict(lambda: defaultdict(list))  # city -> month0 -> [err,...]
    for city, mean_v, settled_v, mon in rows:
        if city is None or mean_v is None or settled_v is None or not mon:
            continue
        err = float(settled_v) - float(mean_v)
        errors[city][mon - 1].append(err)

    bias = {}
    for city, by_month in errors.items():
        bias[city] = {}
        for m0, errs in by_month.items():
            n = len(errs)
            mean_err = sum(errs) / n
            bias[city][m0] = (round(mean_err, 2), n)
    return bias, len(rows)


def climatology_from_historical(cur):
    """
    Reference only: mean actual high per city per month from historical_highs.
    Does NOT give model bias (no stored forecast), but flags data presence.
    Returns clim[city][month0] = (mean_high, n_days)
    """
    cols = describe_table(cur, "historical_highs")
    if cols is None:
        print("\n[historical_highs] table not found — skipping climatology.")
        return {}

    city_col = pick_column(cols, "station_code", "city_code", "city", "code")
    high_col = pick_column(cols, "actual_high", "daily_high", "high", "temp_high")
    date_col = pick_column(cols, "obs_date", "date", "day")
    if not all([city_col, high_col, date_col]):
        print("\n[historical_highs] could not resolve columns — skipping.")
        return {}

    cur.execute(f"""
        SELECT {city_col},
               EXTRACT(MONTH FROM {date_col}::timestamp)::int AS mon,
               AVG({high_col})::float,
               COUNT(*)
        FROM historical_highs
        WHERE {high_col} IS NOT NULL
        GROUP BY {city_col}, mon
    """)
    clim = defaultdict(dict)
    for city, mon, avg_high, n in cur.fetchall():
        if mon:
            clim[city][mon - 1] = (round(avg_high, 1), int(n))
    return clim


def build_new_table(derived):
    """Merge derived bias into existing table; keep hand value when thin."""
    new_table = {c: list(v) for c, v in EXISTING_CITY_BIAS_F.items()}
    notes = []

    for city, by_month in derived.items():
        if city not in new_table:
            new_table[city] = list(DEFAULT_BIAS)
        for m0, (mean_err, n) in by_month.items():
            old_val = new_table[city][m0]
            if n >= MIN_SAMPLES_PER_MONTH:
                new_table[city][m0] = mean_err
                strength = "STRONG" if n >= STRONG_SAMPLES else "ok"
                notes.append(f"  {city:4s} {MONTHS[m0]}: {old_val:+.1f} -> "
                             f"{mean_err:+.2f}  (n={n}, {strength})")
            else:
                notes.append(f"  {city:4s} {MONTHS[m0]}: kept {old_val:+.1f} "
                             f"(only n={n}, need {MIN_SAMPLES_PER_MONTH})")
    return new_table, notes


def print_table(new_table):
    print("\n" + "=" * 70)
    print("CITY_BIAS_F = {")
    for city in sorted(new_table.keys()):
        vals = new_table[city]
        body = ", ".join(f"{v:5.2f}" for v in vals)
        print(f'    "{city}": [{body}],')
    print("}")
    print("=" * 70)


def main():
    conn = connect()
    cur = conn.cursor()
    print("=" * 70)
    print("BIAS DERIVATION — reading logged outcomes (read-only)")
    print("=" * 70)

    derived, n_pred = derive_from_predictions(cur)
    clim = climatology_from_historical(cur)

    if not derived:
        print("\nNo usable prediction rows. Nothing to derive.")
        conn.close()
        return

    print(f"\n[predictions] {n_pred} settled rows analyzed across "
          f"{len(derived)} cities.")

    # Per-city detail, focused on the current month signal
    print("\n--- Measured bias  (settled - corrected_mean) ---")
    print("    +value = model ran COLD (needs +bias).  -value = model ran HOT.")
    for city in sorted(derived.keys()):
        line = [f"  {city:4s}"]
        for m0 in range(12):
            if m0 in derived[city]:
                err, n = derived[city][m0]
                line.append(f"{MONTHS[m0]}:{err:+.1f}(n{n})")
        clim_note = ""
        # show June climatology if present, just as a reference anchor
        if city in clim and 5 in clim[city]:
            avg, nd = clim[city][5]
            clim_note = f"   [hist Jun avg {avg}F, {nd}d]"
        print(" ".join(line) + clim_note)

    new_table, notes = build_new_table(derived)
    print("\n--- Changes applied (>= "
          f"{MIN_SAMPLES_PER_MONTH} samples overwrites; else kept) ---")
    for n in notes:
        print(n)

    print_table(new_table)

    print("\nNEXT STEP:")
    print("  Copy the CITY_BIAS_F block above into the bot, replacing the old")
    print("  table. Re-run this weekly as more bets settle — the values will")
    print("  sharpen and more (city, month) cells will cross the sample floor.")
    conn.close()


if __name__ == "__main__":
    main()
