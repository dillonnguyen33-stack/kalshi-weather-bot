"""
upgrade_to_v3.3.py — Run in the same folder as kalshi_bot_v3.2.py

What it fixes:
  log_prediction() was called before ev_data was computed, so best_side
  and taker_ev were always null in the database. This patch moves the
  log call to after ev_data is computed and passes both fields through.

Usage:
  python3 upgrade_to_v3.3.py
  → produces kalshi_bot_v3.3.py
"""

import sys, os, ast

SRC = "kalshi_bot_v3.2.py"
DST = "kalshi_bot_v3.3.py"

if not os.path.exists(SRC):
    print(f"ERROR: {SRC} not found in current directory ({os.getcwd()})")
    sys.exit(1)

content = open(SRC).read()
original_len = len(content)

# ── PATCH 1: Remove the early log_prediction call (before ev_data exists) ────

OLD1 = """    ev_data   = compute_ev_kelly(prob, market["yes_price"], market["no_price"])
    best      = ev_data["best_side"]

    # Log every prediction for bias calibration — regardless of whether we alert
    with _lock:
        obs_high = asos_observed.get(cc)
    if BIAS_LOGGING:
        try:
            _log_prediction(
                today, cc, CITY_COORDS[cc][2], market["ticker"], threshold,
                forecast, prob, market["yes_price"], market["no_price"], obs_high,
            )
        except Exception as _e:
            print(f"[bias_logger] {_e}")

    if not best: return None"""

NEW1 = """    ev_data   = compute_ev_kelly(prob, market["yes_price"], market["no_price"])
    best      = ev_data["best_side"]

    if not best: return None"""

# ── PATCH 2: Add log call after ev_data is fully computed, with all fields ───

OLD2 = """    with _lock:
        obs_high = asos_observed.get(cc)
    tweet_hit = bool(tweet_cities and cc in tweet_cities)
    afd_hit   = bool(afd_cities and cc in afd_cities)

    fire  = t_ev >= FIRE_EV_THRESHOLD*100 and spread <= MAX_SPREAD_FIRE
    watch = t_ev >= WATCH_EV_THRESHOLD*100 and spread <= MAX_SPREAD_WATCH"""

NEW2 = """    with _lock:
        obs_high = asos_observed.get(cc)
    tweet_hit = bool(tweet_cities and cc in tweet_cities)
    afd_hit   = bool(afd_cities and cc in afd_cities)

    # Log every alert for bias calibration — now with best_side and taker_ev
    if BIAS_LOGGING:
        try:
            _log_prediction(
                today, cc, CITY_COORDS[cc][2], market["ticker"], threshold,
                forecast, prob, market["yes_price"], market["no_price"],
                obs_high,
                best_side=best,
                taker_ev=ev_data[best]["taker_ev"],
                threshold_kind=market["threshold_kind"],
            )
        except Exception as _e:
            print(f"[bias_logger] {_e}")

    fire  = t_ev >= FIRE_EV_THRESHOLD*100 and spread <= MAX_SPREAD_FIRE
    watch = t_ev >= WATCH_EV_THRESHOLD*100 and spread <= MAX_SPREAD_WATCH"""

# ── Apply ─────────────────────────────────────────────────────────────────────

errors = []

if OLD1 not in content:
    errors.append("PATCH 1 target (early log_prediction block) not found")
else:
    content = content.replace(OLD1, NEW1)
    print("✓ Patch 1: removed early log_prediction call")

if OLD2 not in content:
    errors.append("PATCH 2 target (obs_high / tweet_hit block) not found")
else:
    content = content.replace(OLD2, NEW2)
    print("✓ Patch 2: added log_prediction after ev_data with best_side + taker_ev")

# Version bump
content = content.replace(
    "Kalshi Weather Temperature Bot — v3.2",
    "Kalshi Weather Temperature Bot — v3.3"
)
content = content.replace(
    'print("🌡️  Kalshi Weather Bot v3.2")',
    'print("🌡️  Kalshi Weather Bot v3.3")'
)
content = content.replace(
    'print(f"   Upgrades: v3.2 CDF bucket probabilities | fee-adjusted EV | ASOS real-time | bias correction | longshot | maker mode | AFD parser")',
    'print(f"   Upgrades: v3.3 bias_logger best_side+ev fix | v3.2 CDF bucket probabilities | fee-adjusted EV | ASOS real-time | bias correction")'
)
print("✓ Patch 3: version bumped to v3.3")

if errors:
    print("\n⚠️  ERRORS:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)

try:
    ast.parse(content)
    print("✓ Syntax check passed")
except SyntaxError as e:
    print(f"✗ Syntax error: {e}")
    sys.exit(1)

with open(DST, "w") as f:
    f.write(content)

print(f"\n✅ Done — wrote {DST}")
print(f"   {original_len:,} → {len(content):,} chars")
print()
print("What changed:")
print("  • log_prediction() now called AFTER ev_data is computed")
print("  • best_side and taker_ev passed through to predictions.db")
print("  • threshold_kind passed through (T vs B market type)")
print("  • calibration_report() will now show accurate EV columns")
