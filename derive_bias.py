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

# ── Sample thresholds and damping ────────────────────────────────────────────
# Floor lowered to 3 so the current problem cities (LV/PHX/DAL/ATL/HOU/AUS,
# all n>=3 in June) get corrected NOW instead of waiting weeks. To protect
# against a thin-sample average being off, we apply only a FRACTION of the
# measured bias (damping), and we damp harder when samples are fewer.
MIN_SAMPLES_PER_MONTH = 3      # below this, keep the existing hand value
STRONG_SAMPLES        = 15     # at/above this, apply (almost) the full bias

# Damping: applied_bias = measured_bias * damp_factor(n)
#   Few samples  -> trust the measurement less -> smaller fraction applied.
#   Many samples -> trust it more -> closer to full.
# This is deliberately conservative: half-correcting a real +2.5F cold bias
# still removes most of the 50%-hit-rate problem while not over-betting on
# noise. As n grows on each weekly re-run, the correction sharpens.
def damp_factor(n: int) -> float:
    if n >= STRONG_SAMPLES: return 0.85   # strong data, apply most of it
    if n >= 8:              return 0.65
    if n >= 5:              return 0.55
    return 0.50                            # 3-4 samples: apply half

# A measured bias bigger than this is treated as implausible (wrong column,
# bad settle) and rejected rather than written.
MAX_PLAUSIBLE_BIAS_F  = 15.0

# Rolling window (days). 0 = all-time (default). Set via --window N for
# seasonal tracking once each month has enough recent samples to stand alone.
WINDOW_DAYS = 0

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
    # IMPORTANT: actual_high_f is the SETTLED TEMPERATURE. Do NOT use the
    # column literally named "settled" — that is a 0/1 settled-yet FLAG.
    city_col    = pick_column(cols, "city_code", "city", "code")
    actual_col  = pick_column(cols, "actual_high_f", "settled_high",
                              "actual_high", "final_high", "obs_high")
    # The bot bet on the CORRECTED mean = ensemble_mean + bias_applied.
    # We compare the outcome against that, so the new bias replaces the old.
    ens_col     = pick_column(cols, "ensemble_mean", "model_mean", "mean")
    bias_col    = pick_column(cols, "bias_applied")
    # Flag that tells us a row has actually settled.
    settled_flag = pick_column(cols, "settled")
    date_col    = pick_column(cols, "market_date", "target_date", "date",
                              "logged_at", "created_at")

    missing = [n for n, c in [("city", city_col),
                              ("actual_high_f", actual_col),
                              ("ensemble_mean", ens_col),
                              ("date", date_col)]
               if c is None]
    if missing:
        print(f"\n[predictions] Missing required column(s): {missing}")
        print("Edit pick_column() candidates to match your schema, then rerun.")
        return {}, None

    bias_expr = f"COALESCE({bias_col}, 0)" if bias_col else "0"
    flag_clause = f"AND {settled_flag} = 1" if settled_flag else ""
    # Optional rolling window: only rows within the last WINDOW_DAYS.
    window_clause = ""
    if WINDOW_DAYS and WINDOW_DAYS > 0:
        window_clause = (f"AND {date_col}::timestamp "
                         f">= NOW() - INTERVAL '{int(WINDOW_DAYS)} days'")
    print(f"\n[predictions] using: city={city_col} actual={actual_col} "
          f"mean=({ens_col} + {bias_expr}) date={date_col} "
          f"settled_flag={settled_flag or 'none'} "
          f"window={'all-time' if not WINDOW_DAYS else str(WINDOW_DAYS)+'d'}")

    # Only rows that have actually settled (settled flag = 1) and have a
    # real settled high temperature recorded.
    cur.execute(f"""
        SELECT {city_col},
               ({ens_col} + {bias_expr}) AS corrected_mean,
               {actual_col} AS actual_high,
               EXTRACT(MONTH FROM {date_col}::timestamp)::int AS mon
        FROM predictions
        WHERE {actual_col} IS NOT NULL
          AND {ens_col}    IS NOT NULL
          {flag_clause}
          {window_clause}
    """)
    rows = cur.fetchall()

    errors = defaultdict(lambda: defaultdict(list))  # city -> month0 -> [err,...]
    skipped = 0
    for city, corrected_mean, actual, mon in rows:
        if city is None or corrected_mean is None or actual is None or not mon:
            continue
        # Sanity guard: a real high temp is roughly -50..140F. If actual is 0/1
        # we picked the wrong column — refuse rather than emit garbage.
        if not (-50.0 <= float(actual) <= 140.0):
            skipped += 1
            continue
        err = float(actual) - float(corrected_mean)
        errors[city][mon - 1].append(err)

    if skipped:
        print(f"[predictions] WARNING: skipped {skipped} rows with out-of-range "
              f"actual_high (column may be wrong) — investigate if large.")

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
    """Merge derived bias into existing table; keep hand value when thin.

    For cells with enough samples we apply a DAMPED fraction of the measured
    bias rather than the raw value, blending toward the existing value:
        new = old + (measured - old) * damp_factor(n)
    With damp_factor ~0.5 at low n, this moves the cell HALFWAY from the old
    (often wrong-signed) value toward the measured one — enough to fix the
    cold skew, conservative enough to survive a noisy small sample.
    """
    new_table = {c: list(v) for c, v in EXISTING_CITY_BIAS_F.items()}
    notes = []

    for city, by_month in derived.items():
        if city not in new_table:
            new_table[city] = list(DEFAULT_BIAS)
        for m0, (mean_err, n) in by_month.items():
            old_val = new_table[city][m0]
            if abs(mean_err) > MAX_PLAUSIBLE_BIAS_F:
                notes.append(f"  {city:4s} {MONTHS[m0]}: REJECTED {mean_err:+.1f} "
                             f"(implausible, kept {old_val:+.1f}) — check columns")
                continue
            if n >= MIN_SAMPLES_PER_MONTH:
                d = damp_factor(n)
                applied = round(old_val + (mean_err - old_val) * d, 2)
                new_table[city][m0] = applied
                strength = ("STRONG" if n >= STRONG_SAMPLES
                            else "ok" if n >= 5 else "thin")
                notes.append(f"  {city:4s} {MONTHS[m0]}: {old_val:+.1f} -> "
                             f"{applied:+.2f}  (measured {mean_err:+.1f}, "
                             f"n={n}, damp={d:.2f}, {strength})")
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
    global WINDOW_DAYS
    # Parse --window N (days). Default stays all-time.
    args = sys.argv[1:]
    if "--window" in args:
        try:
            WINDOW_DAYS = int(args[args.index("--window") + 1])
        except (ValueError, IndexError):
            print("Usage: python3 derive_bias.py [--window DAYS]")
            sys.exit(1)

    conn = connect()
    cur = conn.cursor()
    print("=" * 70)
    print("BIAS DERIVATION — reading logged outcomes (read-only)")
    if WINDOW_DAYS:
        print(f"  Rolling window: last {WINDOW_DAYS} days")
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
