"""
Kalshi Weather Temperature Bot — v3.34

Changes from v3.33:
  v3.34 — Cross-scan duplicate fix:
           The same ticker was alerting twice in one night at different
           prices (e.g. Seattle B73.5 at 9pm and 11:32pm) because the
           dedup alert-key was anchored on 'today' (ET calendar date),
           which rolled over at ET midnight and minted a fresh key for a
           bet that was really the same settlement market. Alert keys and
           the DB dedup query are now anchored on the SETTLEMENT date
           (target_date) and the afternoon re-alert window is tightened to
           2pm-8pm local. One ticker → one alert per settlement date (plus
           at most one afternoon re-alert).

  (v3.33 — overnight ASOS display fix; v3.32 — Dev tier Push-all +
   Model Accuracy auto-weight; v3.31 — liquidity fix + categories)
"""

import os, asyncio, aiohttp, math, json, time, threading, requests, re

try:
    from bias_logger import log_prediction as _log_prediction
    BIAS_LOGGING = True
except ImportError as e:
    print(f"[startup] bias_logger import FAILED: {e}")
    BIAS_LOGGING = False

print(f"[startup] BIAS_LOGGING={BIAS_LOGGING}")
if BIAS_LOGGING:
    try:
        from bias_logger import ensure_schema
        ensure_schema()
        print("[startup] PostgreSQL connection OK")
    except Exception as e:
        print(f"[startup] PostgreSQL connection FAILED: {e}")
        BIAS_LOGGING = False

from datetime import datetime, date, timezone, timedelta
from zoneinfo import ZoneInfo
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
KALSHI_BASE         = "https://external-api.kalshi.com/trade-api/v2"
OPEN_METEO_BASE     = "https://api.open-meteo.com/v1"
OPEN_METEO_ENS_BASE = "https://ensemble-api.open-meteo.com/v1"
NWS_API_BASE        = "https://api.weather.gov"
AWC_METAR_BASE      = "https://aviationweather.gov/api/data/metar"

ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
TOMORROW_IO_KEY     = os.environ.get("TOMORROW_IO_KEY", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
DISCORD_LOG_WEBHOOK = os.environ.get("DISCORD_LOG_WEBHOOK", "")
WETHR_API_KEY       = os.environ.get("WETHR_API_KEY", "")

SCAN_INTERVAL_SECS  = 300
ASOS_POLL_SECS      = 600
AFD_POLL_SECS       = 3600
FIRE_EV_THRESHOLD   = 0.35
WATCH_EV_THRESHOLD  = 0.20
YES_EV_THRESHOLD    = 0.20  # v3.19: lower threshold since ASOS does the work
MAX_SPREAD_FIRE     = 3.0
MAX_SPREAD_WATCH    = 5.0
MAX_SPREAD_YES      = 2.0   # v3.19: tighter spread for YES — need precision
MAX_CONCURRENT      = 8
BUCKET_GAP_F        = 3.0

ET_TZ  = ZoneInfo("America/New_York")
PT_TZ  = ZoneInfo("America/Los_Angeles")
CT_TZ  = ZoneInfo("America/Chicago")
MT_TZ  = ZoneInfo("America/Denver")
NWS_UA = "KalshiWeatherBot/3.22 dillonnguyen33@gmail.com"

# ── HEATING PROFILES (from PostgreSQL) ───────────────────────────────────────
_heating_profiles_cache: dict = {}
_heating_profiles_loaded = False

def load_heating_profiles_from_db() -> dict:
    """Load heating profiles from PostgreSQL historical data."""
    global _heating_profiles_cache, _heating_profiles_loaded
    if _heating_profiles_loaded:
        return _heating_profiles_cache

    DATABASE_URL = os.environ.get("DATABASE_URL", "")
    if not DATABASE_URL:
        print("[heating] No DATABASE_URL — falling back to JSON")
        return _load_heating_profiles_json()

    try:
        import psycopg2
        conn = psycopg2.connect(DATABASE_URL)
        cur  = conn.cursor()
        cur.execute("""
            SELECT station_code, month,
                   avg_peak_hour, avg_rise_2pm, avg_rise_3pm,
                   avg_rise_4pm, avg_rise_5pm,
                   pct_rising_2pm, pct_rising_3pm,
                   pct_rising_4pm, pct_rising_5pm,
                   days_analyzed
            FROM heating_profiles_db
            ORDER BY station_code, month
        """)
        rows = cur.fetchall()
        conn.close()

        profiles = {}
        for row in rows:
            (station, month, avg_peak, rise_2, rise_3, rise_4, rise_5,
             pct_2, pct_3, pct_4, pct_5, days) = row

            # Map station code to city code
            city_code = WETHR_STATION_TO_CITY.get(station, station)
            if city_code not in profiles:
                profiles[city_code] = {"monthly_profiles": {}}

            profiles[city_code]["monthly_profiles"][month] = {
                "avg_peak_hour":        avg_peak,
                "avg_rise_after_2pm":   rise_2,
                "avg_rise_after_3pm":   rise_3,
                "avg_rise_after_4pm":   rise_4,
                "avg_rise_after_5pm":   rise_5,
                "pct_rising_after_2pm": pct_2,
                "pct_rising_after_3pm": pct_3,
                "pct_rising_after_4pm": pct_4,
                "pct_rising_after_5pm": pct_5,
                "days_analyzed":        days,
            }

        _heating_profiles_cache  = profiles
        _heating_profiles_loaded = True
        print(f"[heating] Loaded real profiles for {len(profiles)} cities from PostgreSQL")
        return profiles

    except Exception as e:
        print(f"[heating] DB load failed: {e} — falling back to JSON")
        return _load_heating_profiles_json()

def _load_heating_profiles_json() -> dict:
    """Fallback to JSON file if DB unavailable."""
    try:
        with open("heating_profiles.json") as f:
            data = json.load(f)
            print(f"[heating] Loaded fallback profiles for {len(data)} cities from JSON")
            return data
    except Exception as e:
        print(f"[heating] Could not load heating_profiles.json: {e}")
        return {}

def get_projected_rise(city_code: str, local_hour: int) -> float:
    """Returns expected additional temp rise from current hour to peak."""
    profiles = load_heating_profiles_from_db()
    month    = datetime.now().month
    profile  = profiles.get(city_code, {})
    mp       = profile.get("monthly_profiles", {}).get(month, {})
    if not mp:
        return 1.0
    if local_hour <= 14:
        return mp.get("avg_rise_after_2pm", 1.0)
    elif local_hour <= 15:
        return mp.get("avg_rise_after_3pm", 0.5)
    elif local_hour <= 16:
        return mp.get("avg_rise_after_4pm", 0.3)
    elif local_hour <= 17:
        return mp.get("avg_rise_after_5pm", 0.1)
    else:
        return 0.05

def get_heating_rate_flag(city_code: str, current_hour: int, current_obs: float) -> float:
    """
    Compares today's morning heating against the historical profile to detect
    if today is running HOT or COLD vs a normal day.

    Returns an extra margin (in °F) to require for pace-confirmation:
      0.0  = today is tracking normal, no extra margin needed
      +1.0 = today is running hot, require more margin (be cautious)
      -0.5 = today is running cool, can be slightly more aggressive

    Logic: we know peak typically occurs around avg_peak_hour and the rise
    amounts after 2/3/4pm. We back out the "expected temp at current hour" from
    the profile, then compare to the actual obs. If actual is well above
    expected, the day is running hot and will likely overshoot the profile.
    """
    profiles = load_heating_profiles_from_db()
    month    = datetime.now().month
    mp       = profiles.get(city_code, {}).get("monthly_profiles", {}).get(month, {})
    if not mp:
        return 0.0

    with _lock:
        curve = dict(asos_intraday.get(city_code, {}))

    # Need at least a morning reading to compare slope
    morning_temps = {h: t for h, t in curve.items() if 7 <= h <= 11}
    if len(morning_temps) < 1:
        return 0.0  # not enough morning data yet

    # Estimate how much the temp has risen from morning to now
    earliest_hr   = min(morning_temps.keys())
    earliest_temp = morning_temps[earliest_hr]
    observed_climb = current_obs - earliest_temp
    hours_elapsed  = max(1, current_hour - earliest_hr)
    climb_per_hour = observed_climb / hours_elapsed

    # Typical climb per hour in this window: derive from profile.
    # avg_rise_after_2pm is what's LEFT after 2pm, so total daytime climb is
    # larger. We use a rough normal of ~2.0F/hr late morning as baseline,
    # adjusted by how "steep" this city's profile is.
    typical_climb_per_hour = 2.0
    if mp.get("avg_rise_after_2pm", 0) > 3.0:
        typical_climb_per_hour = 2.5  # steep-heating city (desert/inland)
    elif mp.get("avg_rise_after_2pm", 0) < 1.0:
        typical_climb_per_hour = 1.2  # flat-heating city (coastal)

    divergence = climb_per_hour - typical_climb_per_hour

    if divergence > 1.0:
        return 1.0   # running notably hot — require +1F more margin
    elif divergence > 0.5:
        return 0.5
    elif divergence < -0.8:
        return -0.5  # running cool — can be slightly more aggressive
    return 0.0

def fetch_dewpoint_depression(city_code: str) -> float | None:
    """
    Fetch current temperature-dewpoint spread (dryness) from Open-Meteo.
    A large spread = dry air = more heating potential than the profile average.
    Returns the depression in °F, or None.
    """
    info = CITY_COORDS.get(city_code)
    if not info:
        return None
    lat, lon = info[0], info[1]
    try:
        r = requests.get(f"{OPEN_METEO_BASE}/forecast", params={
            "latitude": lat, "longitude": lon,
            "current": "temperature_2m,dew_point_2m",
            "temperature_unit": "fahrenheit",
        }, timeout=8)
        r.raise_for_status()
        cur = r.json().get("current", {})
        t  = cur.get("temperature_2m")
        dp = cur.get("dew_point_2m")
        if t is not None and dp is not None:
            return round(float(t) - float(dp), 1)
        return None
    except Exception as e:
        print(f"[dewpoint] {city_code}: {e}")
        return None

def get_dryness_flag(city_code: str) -> float:
    """
    Returns extra margin (°F) to require for pace-confirmation based on dryness.
    Very dry air heats faster/higher than the profile average assumes.
      0.0  = normal humidity
      +0.5 to +1.0 = unusually dry, require more margin
    """
    depression = fetch_dewpoint_depression(city_code)
    if depression is None:
        return 0.0
    # Dew point depression thresholds (°F):
    #   <20F  = humid/normal
    #   20-30F = moderately dry
    #   >30F  = very dry (desert-like), strong extra heating potential
    if depression > 35:
        return 1.0
    elif depression > 25:
        return 0.5
    return 0.0

# ── AFD PRE-FILTER KEYWORDS ───────────────────────────────────────────────────
AFD_ROUTINE_PHRASES = [
    "no significant", "no significant changes",
    "near normal", "close to normal", "around normal",
    "seasonal temperatures", "typical for this time",
    "quiet pattern", "tranquil", "uneventful",
    "high pressure", "dry and sunny", "dry weather",
    "dominated by high pressure",
]
AFD_SIGNAL_KEYWORDS = [
    "uncertainty", "uncertain",
    "warmer than", "cooler than",
    "above normal", "below normal",
    "well above", "well below",
    "warmer than expected", "cooler than expected",
    "model disagreement", "model spread", "model differences",
    "pattern change", "significant change",
    "temperature forecast", "high temperature concern",
    "confidence", "low confidence", "high confidence",
    "degrees warmer", "degrees cooler",
    "surge", "anomaly", "anomalous",
    "record", "exceptional",
]

def afd_should_classify(text: str) -> tuple[bool, str]:
    opening = text[:300].lower()
    for phrase in AFD_ROUTINE_PHRASES:
        if phrase in opening:
            return False, f"routine opening: '{phrase}'"
    body = text.lower()
    for kw in AFD_SIGNAL_KEYWORDS:
        if kw in body:
            return True, f"signal keyword: '{kw}'"
    return False, "no signal keywords found"


# ── CITY CONFIG ───────────────────────────────────────────────────────────────
CITY_COORDS = {
    "NY":  (40.7128,  -74.0060, "New York City",    "KNYC", "OKX", ET_TZ),
    "NYC": (40.7128,  -74.0060, "New York City",    "KNYC", "OKX", ET_TZ),
    "AUS": (30.2672,  -97.7431, "Austin",           "KAUS", "EWX", CT_TZ),
    "LAX": (34.0522, -118.2437, "Los Angeles",      "KLAX", "LOX", PT_TZ),
    "CHI": (41.8781,  -87.6298, "Chicago",          "KMDW", "LOT", CT_TZ),
    "MIA": (25.7617,  -80.1918, "Miami",            "KMIA", "MFL", ET_TZ),
    "DAL": (32.7767,  -96.7970, "Dallas",           "KDFW", "FWD", CT_TZ),
    "DC":  (38.9072,  -77.0369, "Washington DC",    "KDCA", "LWX", ET_TZ),
    "SEA": (47.6062, -122.3321, "Seattle",          "KSEA", "SEW", PT_TZ),
    "PHX": (33.4484, -112.0740, "Phoenix",          "KPHX", "PSR", MT_TZ),
    "BOS": (42.3601,  -71.0589, "Boston",           "KBOS", "BOX", ET_TZ),
    "HOU": (29.7604,  -95.3698, "Houston",          "KHOU", "HGX", CT_TZ),
    "ATL": (33.7490,  -84.3880, "Atlanta",          "KATL", "FFC", ET_TZ),
    "OKC": (35.4676,  -97.5164, "Oklahoma City",    "KOKC", "OUN", CT_TZ),
    "LV":  (36.1699, -115.1398, "Las Vegas",        "KLAS", "VEF", PT_TZ),
    "SFO": (37.7749, -122.4194, "San Francisco",    "KSFO", "MTR", PT_TZ),
    "DEN": (39.7392, -104.9903, "Denver",           "KDEN", "BOU", MT_TZ),
    "SA":  (29.4241,  -98.4936, "San Antonio",      "KSAT", "EWX", CT_TZ),
    "NO":  (29.9511,  -90.0715, "New Orleans",      "KMSY", "LIX", CT_TZ),
    "MN":  (44.9778,  -93.2650, "Minneapolis",      "KMSP", "MPX", CT_TZ),
    "PHI": (39.9526,  -75.1652, "Philadelphia",     "KPHL", "PHI", ET_TZ),
    "MEM": (35.1495,  -90.0490, "Memphis",          "KMEM", "MEG", CT_TZ),
    "PI":  (40.4406,  -79.9959, "Pittsburgh",       "KPIT", "PBZ", ET_TZ),
    "BA":  (39.2904,  -76.6122, "Baltimore",        "KBWI", "LWX", ET_TZ),
    "CL":  (41.4993,  -81.6944, "Cleveland",        "KCLE", "CLE", ET_TZ),
    "SD":  (32.7157, -117.1611, "San Diego",        "KSAN", "SGX", PT_TZ),
    "KC":  (39.0997,  -94.5786, "Kansas City",      "KMCI", "EAX", CT_TZ),
    "SL":  (38.6270,  -90.1994, "St. Louis",        "KSTL", "LSX", CT_TZ),
    "PO":  (45.5051, -122.6750, "Portland",         "KPDX", "PQR", PT_TZ),
    "AL":  (35.2220,  -80.8431, "Charlotte",        "KCLT", "GSP", ET_TZ),
    "IN":  (39.7684,  -86.1581, "Indianapolis",     "KIND", "IND", ET_TZ),
    "COL": (39.9612,  -82.9988, "Columbus",         "KCMH", "ILN", ET_TZ),
    "TUC": (32.2226, -110.9747, "Tucson",           "KTUS", "TWC", MT_TZ),
    "EL":  (31.7619, -106.4850, "El Paso",          "KELP", "EPZ", MT_TZ),
    "MIL": (43.0389,  -87.9065, "Milwaukee",        "KMKE", "MKX", CT_TZ),
    "RAL": (35.7796,  -78.6382, "Raleigh",          "KRDU", "RAH", ET_TZ),
    "TAM": (27.9506,  -82.4572, "Tampa",            "KTPA", "TBW", ET_TZ),
    "SLC": (40.7608, -111.8910, "Salt Lake City",   "KSLC", "SLC", MT_TZ),
    "OL":  (36.1627,  -86.7816, "Nashville",        "KBNA", "OHX", CT_TZ),
    "DE":  (42.3314,  -83.0458, "Detroit",          "KDTW", "DTX", ET_TZ),
}

# ── BIAS CORRECTIONS ──────────────────────────────────────────────────────────
CITY_BIAS_F = {
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

def get_bias(city_code: str) -> float:
    month = datetime.now().month - 1
    return CITY_BIAS_F.get(city_code, DEFAULT_BIAS)[month]

def get_city_tz(city_code: str) -> ZoneInfo:
    info = CITY_COORDS.get(city_code)
    if info and len(info) >= 6:
        return info[5]
    return ET_TZ

def get_bet_category(city_code: str, is_next_day: bool, pace_confirmed: bool = False) -> str:
    """
    Categorize a bet by city local time of day.
      overnight       = 8pm-6am
      morning         = 6am-12pm
      pacing          = 12pm-8pm (afternoon, not trajectory-confirmed)
      pace_confirmed  = afternoon bet where observed trajectory confirms it
    pace_confirmed is its own category so the scoreboard can separate the
    highest-confidence bets from regular afternoon ones.
    """
    if pace_confirmed:
        return "pace_confirmed"
    hour = datetime.now(get_city_tz(city_code)).hour
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 20:
        return "pacing"
    else:
        return "overnight"

# Discord embed colors by category
CATEGORY_COLORS = {
    "overnight":      0x5865F2,  # indigo
    "morning":        0x1a6b3a,  # green
    "pacing":         0xe67e22,  # orange
    "pace_confirmed": 0xf1c40f,  # gold
}
CATEGORY_EMOJI = {
    "overnight":      "🌙",
    "morning":        "🌅",
    "pacing":         "📈",
    "pace_confirmed": "✅",
}


# ── RUNTIME STATE ─────────────────────────────────────────────────────────────
asos_observed: dict[str, float]  = {}
asos_intraday: dict              = {}   # {city: {hour: temp}} for morning heating rate
city_dewpoint: dict[str, float]  = {}   # {city: dewpoint_depression_f}
afd_flagged_cities: set          = set()
seen_afd_ids: set                = set()
posted_alert_keys: set           = set()
last_reset_date                  = None
_lock = threading.Lock()

# ── DAILY RESET ───────────────────────────────────────────────────────────────
def maybe_reset_daily():
    global posted_alert_keys, last_reset_date
    today = datetime.now(ET_TZ).date()
    if last_reset_date is None:
        last_reset_date = today
        return
    if today > last_reset_date:
        print("[reset] New day — clearing posted alert keys + intraday tracking")
        posted_alert_keys.clear()
        with _lock:
            asos_intraday.clear()
        last_reset_date = today

CITY_KEYWORD_MAP = {
    "new york":"NY","nyc":"NY","central park":"NY","manhattan":"NY",
    "chicago":"CHI","windy city":"CHI",
    "los angeles":"LAX","socal":"LAX","l.a.":"LAX",
    "miami":"MIA","south florida":"MIA",
    "philadelphia":"PH","philly":"PH",
    "atlanta":"AT","atl":"AT",
    "minneapolis":"MN","twin cities":"MN",
    "san francisco":"SF","bay area":"SF",
    "dallas":"DA","dfw":"DA","fort worth":"DA",
    "boston":"BO","houston":"HO","detroit":"DE","seattle":"SE",
    "phoenix":"PHX","denver":"DEN","las vegas":"LV","san diego":"SD",
    "kansas city":"KC","st. louis":"SL","saint louis":"SL",
    "new orleans":"NO","nola":"NO","cleveland":"CL","pittsburgh":"PI",
    "baltimore":"BA","washington":"DC","d.c.":"DC","nashville":"OL",
    "memphis":"MEM","san antonio":"SA","austin":"AUS","portland":"PO",
    "salt lake":"SLC","charlotte":"AL","indianapolis":"IN","columbus":"COL",
    "oklahoma city":"OK","okc":"OK","tucson":"TUC","el paso":"EL",
    "milwaukee":"MIL","raleigh":"RAL","tampa":"TAM",
}

# ── KALSHI FEE MATH ───────────────────────────────────────────────────────────
def kalshi_taker_fee(price_cents: int) -> float:
    p = price_cents / 100
    return 0.07 * p * (1 - p)

def kalshi_maker_fee(price_cents: int) -> float:
    return kalshi_taker_fee(price_cents) * 0.25

def compute_ev_kelly(model_prob: float, yes_price: int, no_price: int) -> dict:
    results = {}
    for side, price, prob in [("YES", yes_price, model_prob),
                               ("NO",  no_price,  1 - model_prob)]:
        p       = price / 100
        t_fee   = kalshi_taker_fee(price)
        m_fee   = kalshi_maker_fee(price)
        win     = 1 - p
        t_ev    = prob * (win - t_fee) - (1 - prob) * p
        m_ev    = prob * (win - m_fee) - (1 - prob) * p
        t_kelly = max(0, (prob * win - (1 - prob) * p) / win) if win > 0 else 0
        m_kelly = max(0, t_kelly * (1 + (t_fee - m_fee) / win)) if win > 0 else 0
        results[side] = {
            "prob":          round(prob * 100, 1),
            "implied":       round(p * 100, 1),
            "taker_ev":      round(t_ev * 100, 1),
            "maker_ev":      round(m_ev * 100, 1),
            "taker_hk":      round(t_kelly * 0.5 * 100, 1),
            "maker_hk":      round(m_kelly * 0.5 * 100, 1),
            "taker_fee_pct": round(t_fee / win * 100, 1) if win > 0 else 0,
        }
    best = max(("YES", "NO"), key=lambda s: results[s]["taker_ev"])
    if results[best]["taker_ev"] <= 0:
        best = None
    return {"best_side": best, "YES": results["YES"], "NO": results["NO"]}

# ── WETHR.NET OBSERVATIONS ────────────────────────────────────────────────────
WETHR_OBS_BASE = "https://wethr.net/api/v2/observations.php"
WETHR_PUSH_URL = "https://wethr.net:3443/api/v2/stream"

# Map city codes to Wethr station codes
# Only stations supported by Wethr Professional tier
WETHR_STATIONS = {
    "NY":  "KNYC", "AUS": "KAUS", "LAX": "KLAX", "CHI": "KMDW",
    "MIA": "KMIA", "DAL": "KDFW", "DC":  "KDCA", "SEA": "KSEA",
    "PHX": "KPHX", "BOS": "KBOS", "HOU": "KHOU", "ATL": "KATL",
    "OKC": "KOKC", "LV":  "KLAS", "SFO": "KSFO", "DEN": "KDEN",
    "SA":  "KSAT", "MN":  "KMSP", "NO":  "KMSY",
}

# Push API — Developer tier supports ALL stations (was 5 on Professional)
WETHR_PUSH_STATIONS = list(WETHR_STATIONS.values())

# Reverse map: Wethr station → city code (for Push API)
WETHR_STATION_TO_CITY = {v: k for k, v in WETHR_STATIONS.items()}

WETHR_FORECAST_BASE = "https://wethr.net/api/v2/forecasts.php"
WETHR_NWS_BASE      = "https://wethr.net/api/v2/nws_forecasts.php"
WETHR_ACCURACY_BASE = "https://wethr.net/api/v2/model_accuracy.php"  # Developer tier
WETHR_MODELS        = ["HRRR", "NBM", "RAP", "NAM4KM", "GFS", "ECMWF-IFS"]

# Model Accuracy cache — refreshed once per scan cycle, drives auto-weighting
_model_accuracy_cache: dict = {}   # {station: {model: mae_f}}
_model_accuracy_ts: dict    = {}
MODEL_ACCURACY_TTL = 21600  # 6 hours — accuracy stats change slowly

# Cache NWS forecasts to avoid rate limiting (refresh every 2 hours)
_nws_cache: dict = {}
_nws_cache_ts: dict = {}
NWS_CACHE_TTL = 7200  # 2 hours

_wethr_fcst_cache: dict = {}
_wethr_fcst_ts: dict   = {}
WETHR_FCST_TTL = 5400  # 90 min cache — models only update every 1-6 hours

def fetch_wethr_forecast_high(station: str, model: str, target_date: str) -> float | None:
    """Fetch forecast high from Wethr using latest model run — cached."""
    if not WETHR_API_KEY:
        return None
    cache_key = f"{station}_{model}_{target_date}"
    now = time.time()
    if cache_key in _wethr_fcst_cache and now - _wethr_fcst_ts.get(cache_key, 0) < WETHR_FCST_TTL:
        return _wethr_fcst_cache[cache_key]
    try:
        r = requests.get(WETHR_FORECAST_BASE, params={
            "location_name": station,
            "model":         model,
            "run":           "latest",
        }, headers={"Authorization": f"Bearer {WETHR_API_KEY}"}, timeout=10)
        if r.status_code == 429:
            time.sleep(10)
            r = requests.get(WETHR_FORECAST_BASE, params={
                "location_name": station,
                "model":         model,
                "run":           "latest",
            }, headers={"Authorization": f"Bearer {WETHR_API_KEY}"}, timeout=10)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            _wethr_fcst_cache[cache_key]  = None
            _wethr_fcst_ts[cache_key]     = now
            return None
        highs = []
        for row in rows:
            if row.get("valid_time", "")[:10] == target_date:
                temp_f = row.get("temperature_f")
                if temp_f is not None:
                    highs.append(float(temp_f))
        result = max(highs) if highs else None
        _wethr_fcst_cache[cache_key] = result
        _wethr_fcst_ts[cache_key]    = now
        return result
    except Exception as e:
        print(f"[wethr_fcst] {station}/{model}: {e}")
        return None

def fetch_wethr_nws_forecast(station: str, target_date: str) -> float | None:
    """Fetch NWS hourly forecast high from Wethr — cached to avoid rate limits."""
    if not WETHR_API_KEY:
        return None
    cache_key = f"{station}_{target_date}"
    now = time.time()
    if cache_key in _nws_cache and now - _nws_cache_ts.get(cache_key, 0) < NWS_CACHE_TTL:
        return _nws_cache[cache_key]
    try:
        r = requests.get(WETHR_NWS_BASE, params={
            "station_code": station,
            "date":         target_date,
            "mode":         "latest",
        }, headers={"Authorization": f"Bearer {WETHR_API_KEY}"}, timeout=10)
        if r.status_code == 429:
            time.sleep(10)
            r = requests.get(WETHR_NWS_BASE, params={
                "station_code": station,
                "date":         target_date,
                "mode":         "latest",
            }, headers={"Authorization": f"Bearer {WETHR_API_KEY}"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        high = data.get("high")
        result = float(high) if high is not None else None
        _nws_cache[cache_key]    = result
        _nws_cache_ts[cache_key] = now
        return result
    except Exception as e:
        print(f"[wethr_nws] {station}: {e}")
        return None

def fetch_wethr_all_models(station: str, target_date: str, cache_only: bool = False) -> dict:
    """Fetch all Wethr model forecasts for a station and date.
    If cache_only=True, only returns cached values (no API calls)."""
    if not WETHR_API_KEY or not station:
        return {}
    results = {}
    now = time.time()
    for model in WETHR_MODELS:
        cache_key = f"{station}_{model}_{target_date}"
        if cache_only:
            # Only read from cache, never call API
            if cache_key in _wethr_fcst_cache and \
               now - _wethr_fcst_ts.get(cache_key, 0) < WETHR_FCST_TTL:
                val = _wethr_fcst_cache[cache_key]
                if val is not None:
                    results[model] = val
            continue
        high = fetch_wethr_forecast_high(station, model, target_date)
        if high is not None:
            results[model] = high
        time.sleep(1.5)

    # NWS forecast
    nws_key = f"{station}_{target_date}"
    if cache_only:
        if nws_key in _nws_cache and now - _nws_cache_ts.get(nws_key, 0) < NWS_CACHE_TTL:
            val = _nws_cache[nws_key]
            if val is not None:
                results["NWS"] = val
    else:
        nws = fetch_wethr_nws_forecast(station, target_date)
        if nws is not None:
            results["NWS"] = nws
    return results

def fetch_model_accuracy(station: str) -> dict:
    """
    Fetch per-model accuracy (MAE in °F) for a station from Wethr's
    Model Accuracy API (Developer tier). Returns {model: mae_f}.
    Lower MAE = more accurate model. Cached 6h.
    """
    if not WETHR_API_KEY or not station:
        return {}
    now = time.time()
    if station in _model_accuracy_cache and \
       now - _model_accuracy_ts.get(station, 0) < MODEL_ACCURACY_TTL:
        return _model_accuracy_cache[station]
    try:
        r = requests.get(WETHR_ACCURACY_BASE, params={
            "station_code": station,
            "window":       "30d",   # 30-day trailing accuracy
            "metric":       "mae",
        }, headers={"Authorization": f"Bearer {WETHR_API_KEY}"}, timeout=10)
        if r.status_code == 429:
            time.sleep(5)
            r = requests.get(WETHR_ACCURACY_BASE, params={
                "station_code": station, "window": "30d", "metric": "mae",
            }, headers={"Authorization": f"Bearer {WETHR_API_KEY}"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        # Expected: {"models": {"HRRR": {"mae_f": 1.8}, "NBM": {"mae_f": 1.4}, ...}}
        accuracy = {}
        models_obj = data.get("models", {})
        for model, stats in models_obj.items():
            mae = stats.get("mae_f") if isinstance(stats, dict) else None
            if mae is not None:
                accuracy[model] = float(mae)
        _model_accuracy_cache[station] = accuracy
        _model_accuracy_ts[station]    = now
        if accuracy:
            best = min(accuracy, key=accuracy.get)
            print(f"[accuracy] {station}: {len(accuracy)} models, best={best} "
                  f"(MAE {accuracy[best]:.1f}°F)")
        return accuracy
    except Exception as e:
        print(f"[accuracy] {station}: {e}")
        return {}

def accuracy_to_weights(accuracy: dict) -> dict:
    """
    Convert per-model MAE into blend weights. Lower MAE → higher weight.
    Uses inverse-MAE weighting: weight = (1 / mae)^2, normalized so the
    best model gets a meaningful edge but no model is fully dropped.
    Returns {model: weight_multiplier} where multiplier scales how many
    copies of that model go into the blend (clamped 1-6).
    """
    if not accuracy:
        return {}
    weights = {}
    inv = {m: (1.0 / max(mae, 0.3)) ** 2 for m, mae in accuracy.items()}
    max_inv = max(inv.values()) if inv else 1.0
    for model, v in inv.items():
        # Scale to 1-6 copies; best model ~6, worst ~1
        w = 1 + round(5 * (v / max_inv))
        weights[model] = max(1, min(6, w))
    return weights

def calculate_model_pacing(wethr_models: dict, current_wethr_high: float | None) -> dict:
    """Calculate how each model is pacing vs live Wethr High."""
    if current_wethr_high is None:
        return {}
    return {model: round(current_wethr_high - fcast, 1)
            for model, fcast in wethr_models.items() if fcast is not None}

def apply_pacing_correction(mean: float, pacing: dict, local_hour: int) -> float:
    """Apply model pacing to adjust ensemble mean after noon."""
    if not pacing or local_hour < 12:
        return mean
    if local_hour >= 18:   pace_weight = 0.7
    elif local_hour >= 16: pace_weight = 0.5
    elif local_hour >= 14: pace_weight = 0.3
    else:                  pace_weight = 0.15
    pace_vals = list(pacing.values())
    if not pace_vals:
        return mean
    avg_pace = sum(pace_vals) / len(pace_vals)
    if len(pace_vals) >= 3:
        std = (sum((p - avg_pace)**2 for p in pace_vals) / len(pace_vals)) ** 0.5
        pace_vals = [p for p in pace_vals if abs(p - avg_pace) <= 2 * std]
        avg_pace  = sum(pace_vals) / len(pace_vals) if pace_vals else avg_pace
    correction = avg_pace * pace_weight
    corrected  = round(mean + correction, 2)
    if abs(correction) > 0.1:
        print(f"[pacing] {mean}°F + {correction:+.1f}°F → {corrected}°F "
              f"(weight={pace_weight}, pace={avg_pace:+.1f}°F)")
    return corrected

def fetch_wethr_high(city_code: str) -> float | None:
    """Fetch today's confirmed high from Wethr.net using NWS logic (matches Kalshi settlement)."""
    if not WETHR_API_KEY:
        return None
    station = WETHR_STATIONS.get(city_code)
    if not station:
        return None
    try:
        r = requests.get(WETHR_OBS_BASE, params={
            "station_code": station,
            "mode": "wethr_high",
            "logic": "nws",
        }, headers={"Authorization": f"Bearer {WETHR_API_KEY}"}, timeout=8)
        r.raise_for_status()
        data = r.json()
        high = data.get("wethr_high")
        if high is not None:
            return float(high)
        return None
    except Exception as e:
        print(f"[wethr] {city_code}/{station}: {e}")
        return None

def fetch_asos_high_fallback(city_code: str) -> float | None:
    """Fallback to Aviation Weather METAR if Wethr unavailable."""
    info = CITY_COORDS.get(city_code)
    if not info: return None
    icao = info[3]
    try:
        r = requests.get(AWC_METAR_BASE,
            params={"ids": icao, "format": "json", "hours": 14},
            headers={"User-Agent": NWS_UA}, timeout=8)
        r.raise_for_status()
        data = r.json()
        if not data: return None
        temps = [obs.get("temp") for obs in data if obs.get("temp") is not None]
        if not temps: return None
        return round(max(temps) * 9 / 5 + 32, 1)
    except Exception as e:
        print(f"[asos_fallback] {city_code}/{icao}: {e}")
        return None

def fetch_asos_high(city_code: str) -> float | None:
    """Fetch observed high — prefer Wethr.net, fall back to Aviation Weather."""
    high = fetch_wethr_high(city_code)
    if high is not None:
        return high
    return fetch_asos_high_fallback(city_code)

def asos_poll_loop():
    print(f"[asos] Starting observation poll for {len(CITY_COORDS)} cities")
    print(f"[asos] Using {'Wethr.net API' if WETHR_API_KEY else 'Aviation Weather fallback'}")
    while True:
        for code in CITY_COORDS:
            high = fetch_asos_high(code)
            if high is not None:
                with _lock:
                    asos_observed[code] = high
                    # Record intraday curve: store the obs high at each local hour
                    local_hr = datetime.now(get_city_tz(code)).hour
                    if code not in asos_intraday:
                        asos_intraday[code] = {}
                    asos_intraday[code][local_hr] = high
        time.sleep(ASOS_POLL_SECS)

def wethr_push_loop():
    """Listen to Wethr.net Push API for real-time new_high events.
    When a city hits a new confirmed high, immediately rescan that city."""
    if not WETHR_API_KEY:
        print("[push] No WETHR_API_KEY — Push API disabled")
        return

    # Only subscribe to top 5 stations (Professional tier limit)
    stations = ",".join(WETHR_PUSH_STATIONS)
    url = f"{WETHR_PUSH_URL}?stations={stations}&api_key={WETHR_API_KEY}"
    print(f"[push] Connecting to Wethr Push API for {len(WETHR_PUSH_STATIONS)} stations: {stations}")

    while True:
        try:
            r = requests.get(url, stream=True, timeout=300,
                           headers={"Accept": "text/event-stream"})
            r.raise_for_status()

            event_type = None
            for line in r.iter_lines():
                if not line:
                    event_type = None
                    continue
                line = line.decode("utf-8") if isinstance(line, bytes) else line

                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:") and event_type in ("new_high", "observation"):
                    try:
                        data = json.loads(line[5:].strip())
                        station = data.get("station_code", "")
                        city_code = WETHR_STATION_TO_CITY.get(station)

                        if event_type == "new_high" and city_code:
                            new_val = data.get("value_f")
                            if new_val:
                                with _lock:
                                    asos_observed[city_code] = float(new_val)
                                print(f"[push] 🔥 NEW HIGH {city_code}/{station}: {new_val}°F — rescanning")
                                run_scan(force_codes={city_code})

                        elif event_type == "observation" and city_code:
                            # Update current observed high from wethr_high in observation
                            wethr_high = data.get("wethr_high", {}).get("nws", {}).get("value_f")
                            if wethr_high:
                                with _lock:
                                    old = asos_observed.get(city_code, 0)
                                    if float(wethr_high) > old:
                                        asos_observed[city_code] = float(wethr_high)
                    except Exception as e:
                        print(f"[push] Parse error: {e}")

        except Exception as e:
            print(f"[push] Connection error: {e} — reconnecting in 30s")
            time.sleep(30)

# ── NWS AFD PARSER ────────────────────────────────────────────────────────────
def fetch_afd_text(wfo: str) -> str | None:
    try:
        r = requests.get(f"{NWS_API_BASE}/products/types/AFD/locations/{wfo}",
            headers={"User-Agent": NWS_UA, "Accept": "application/geo+json"}, timeout=10)
        r.raise_for_status()
        products = r.json().get("@graph", [])
        if not products: return None
        latest_id = products[0].get("id")
        if not latest_id or latest_id in seen_afd_ids: return None
        seen_afd_ids.add(latest_id)
        r2 = requests.get(f"{NWS_API_BASE}/products/{latest_id}",
            headers={"User-Agent": NWS_UA}, timeout=10)
        r2.raise_for_status()
        return r2.json().get("productText", "")
    except Exception as e:
        print(f"[afd] {wfo}: {e}")
        return None

def classify_afd(text: str, wfo: str) -> dict:
    if not ANTHROPIC_API_KEY or not text:
        return {"is_signal": False, "cities": [], "direction": "", "summary": ""}
    excerpt = text[:1500]
    system = (
        "You classify NWS Area Forecast Discussion (AFD) text for weather market signals. "
        "Respond ONLY with valid JSON. "
        "A SIGNAL = the AFD mentions: model disagreement, uncertain temperature forecast, "
        "pattern change affecting high temps, significant warm/cold deviation from normal, "
        "or forecaster explicitly flagging temperature forecast confidence issues. "
        "Routine stable forecasts are NOT signals. "
        '{"is_signal":bool,"cities":["city names"],"direction":"warmer"|"cooler"|"uncertain"|"",'
        '"confidence":"high"|"medium"|"low","summary":"one sentence or empty string"}'
    )
    try:
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key":ANTHROPIC_API_KEY,"anthropic-version":"2023-06-01",
                     "content-type":"application/json"},
            json={"model":"claude-haiku-4-5-20251001","max_tokens":200,"system":system,
                  "messages":[{"role":"user","content":f"WFO: {wfo}\n\nAFD excerpt:\n{excerpt}"}]},
            timeout=15)
        r.raise_for_status()
        data = r.json()
        if "content" not in data or not data["content"]:
            return {"is_signal": False, "cities": [], "direction": "", "summary": ""}
        return json.loads(data["content"][0]["text"].strip())
    except json.JSONDecodeError:
        return {"is_signal": False, "cities": [], "direction": "", "summary": ""}
    except Exception as e:
        print(f"[claude/afd] {e}")
        return {"is_signal": False, "cities": [], "direction": "", "summary": ""}

def names_to_codes(cities: list[str]) -> list[str]:
    codes = []
    for city in cities:
        cl = city.lower()
        for kw, code in CITY_KEYWORD_MAP.items():
            if kw in cl:
                codes.append(code); break
    return list(set(codes))

def wfo_to_city_codes(wfo: str) -> list[str]:
    mapping = {
        "OKX":["NY"],"LOT":["CHI","MIL"],"LOX":["LAX","SD"],
        "MFL":["MIA","TAM"],"PHI":["PH","BA","DC"],"LWX":["BA","DC"],
        "FFC":["AT"],"MPX":["MN"],"MTR":["SF"],"FWD":["DA"],"BOX":["BO"],
        "HGX":["HO"],"DTX":["DE","CL"],"SEW":["SE"],"PSR":["PHX","TUC"],
        "BOU":["DEN"],"VEF":["LV"],"SGX":["SD"],"EAX":["KC"],"LSX":["SL"],
        "LIX":["NO"],"CLE":["CL","PI"],"PBZ":["PI"],"OHX":["OL"],
        "MEG":["MEM"],"EWX":["SA","AUS"],"PQR":["PO"],"SLC":["SLC"],
        "GSP":["AL","RAL"],"IND":["IN"],"ILN":["COL"],"OUN":["OK"],
        "TWC":["TUC"],"EPZ":["EL"],"MKX":["MIL"],"RAH":["RAL"],"TBW":["TAM"],
    }
    return mapping.get(wfo.upper(), [])

def afd_scanner_loop():
    wfos = list(set(info[4] for info in CITY_COORDS.values()))
    print(f"[afd] Monitoring {len(wfos)} NWS forecast offices")
    afd_total = afd_skipped = afd_classified = 0
    while True:
        for wfo in wfos:
            text = fetch_afd_text(wfo)
            if not text: continue
            afd_total += 1
            should_classify, reason = afd_should_classify(text)
            if not should_classify:
                afd_skipped += 1
                print(f"[afd] {wfo} skipped ({reason}) — {afd_skipped}/{afd_total} filtered")
                continue
            afd_classified += 1
            print(f"[afd] {wfo} → Claude ({reason}) — {afd_classified}/{afd_total} classified")
            clf = classify_afd(text, wfo)
            time.sleep(2)
            if clf.get("is_signal"):
                codes = names_to_codes(clf.get("cities", []))
                if not codes: codes = wfo_to_city_codes(wfo)
                if codes:
                    with _lock: afd_flagged_cities.update(codes)
                    print(f"[afd] Signal {wfo}: {codes} | {clf.get('direction')} | {clf.get('summary')}")
        time.sleep(AFD_POLL_SECS)

# ── ASYNC FORECAST ENGINE ─────────────────────────────────────────────────────
async def _get_json(session: aiohttp.ClientSession, url: str, params: dict) -> dict:
    async with session.get(url, params=params) as r:
        return await r.json()

async def fetch_ecmwf(session, lat, lon, ds) -> float | None:
    try:
        data = await _get_json(session, f"{OPEN_METEO_BASE}/ecmwf", {
            "latitude":lat,"longitude":lon,"hourly":"temperature_2m",
            "temperature_unit":"fahrenheit","start_date":ds,"end_date":ds})
        temps = [t for t in (data.get("hourly",{}).get("temperature_2m") or []) if t is not None]
        return max(temps) if temps else None
    except: return None

async def fetch_aifs(session, lat, lon, ds) -> float | None:
    try:
        data = await _get_json(session, f"{OPEN_METEO_BASE}/ecmwf", {
            "latitude":lat,"longitude":lon,"hourly":"temperature_2m",
            "temperature_unit":"fahrenheit","start_date":ds,"end_date":ds,
            "models":"ecmwf_aifs025"})
        temps = [t for t in (data.get("hourly",{}).get("temperature_2m") or []) if t is not None]
        return max(temps) if temps else None
    except: return None

async def fetch_hrrr(session, lat, lon, ds) -> float | None:
    try:
        data = await _get_json(session, f"{OPEN_METEO_BASE}/gfs", {
            "latitude":lat,"longitude":lon,"hourly":"temperature_2m",
            "temperature_unit":"fahrenheit","start_date":ds,"end_date":ds,"models":"ncep_hrrr_conus"})
        temps = [t for t in (data.get("hourly",{}).get("temperature_2m") or []) if t is not None]
        return max(temps) if temps else None
    except: return None

async def fetch_rap(session, lat, lon, ds) -> float | None:
    return None

async def fetch_gfs_ensemble(session, lat, lon, ds) -> list[float]:
    try:
        members = ",".join([f"temperature_2m_member{i:02d}" for i in range(1,32)])
        data = await _get_json(session, f"{OPEN_METEO_ENS_BASE}/ensemble", {
            "latitude":lat,"longitude":lon,"hourly":members,
            "temperature_unit":"fahrenheit","start_date":ds,"end_date":ds,"models":"gfs_seamless"})
        hourly = data.get("hourly",{})
        highs = []
        for i in range(1,32):
            temps = [t for t in (hourly.get(f"temperature_2m_member{i:02d}") or []) if t is not None]
            if temps: highs.append(max(temps))
        return highs
    except: return []

async def fetch_nbm(session, lat, lon, ds) -> float | None:
    try:
        data = await _get_json(session, f"{OPEN_METEO_BASE}/forecast", {
            "latitude":lat,"longitude":lon,"hourly":"temperature_2m",
            "temperature_unit":"fahrenheit","start_date":ds,"end_date":ds,"models":"ncep_nbm_conus"})
        temps = [t for t in (data.get("hourly",{}).get("temperature_2m") or []) if t is not None]
        return max(temps) if temps else None
    except: return None

async def fetch_nbm_probabilistic(session, lat, lon, ds) -> list[float]:
    try:
        variables = ",".join([
            "temperature_2m_p10","temperature_2m_p25","temperature_2m_p50",
            "temperature_2m_p75","temperature_2m_p90"
        ])
        data = await _get_json(session, f"{OPEN_METEO_BASE}/forecast", {
            "latitude":lat,"longitude":lon,"hourly":variables,
            "temperature_unit":"fahrenheit","start_date":ds,"end_date":ds,"models":"ncep_nbm_conus"})
        hourly = data.get("hourly",{})
        highs = []
        for pct in ["temperature_2m_p10","temperature_2m_p25","temperature_2m_p50",
                    "temperature_2m_p75","temperature_2m_p90"]:
            temps = [t for t in (hourly.get(pct) or []) if t is not None]
            if temps: highs.append(max(temps))
        return highs
    except: return []

async def fetch_icon(session, lat, lon, ds) -> float | None:
    try:
        data = await _get_json(session, f"{OPEN_METEO_BASE}/forecast", {
            "latitude":lat,"longitude":lon,"hourly":"temperature_2m",
            "temperature_unit":"fahrenheit","start_date":ds,"end_date":ds,"models":"icon_seamless"})
        temps = [t for t in (data.get("hourly",{}).get("temperature_2m") or []) if t is not None]
        return max(temps) if temps else None
    except: return None

async def fetch_ecmwf_ensemble(session, lat, lon, ds) -> list[float]:
    try:
        members = ",".join([f"temperature_2m_member{i:02d}" for i in range(0,51)])
        data = await _get_json(session, f"{OPEN_METEO_ENS_BASE}/ensemble", {
            "latitude":lat,"longitude":lon,"hourly":members,
            "temperature_unit":"fahrenheit","start_date":ds,"end_date":ds,"models":"ecmwf_ifs025"})
        hourly = data.get("hourly",{})
        highs = []
        for i in range(0,51):
            temps = [t for t in (hourly.get(f"temperature_2m_member{i:02d}") or []) if t is not None]
            if temps: highs.append(max(temps))
        return highs
    except: return []

async def fetch_icon_ensemble(session, lat, lon, ds) -> list[float]:
    try:
        members = ",".join([f"temperature_2m_member{i:02d}" for i in range(0,40)])
        data = await _get_json(session, f"{OPEN_METEO_ENS_BASE}/ensemble", {
            "latitude":lat,"longitude":lon,"hourly":members,
            "temperature_unit":"fahrenheit","start_date":ds,"end_date":ds,
            "models":"icon_seamless"})
        hourly = data.get("hourly",{})
        highs = []
        for i in range(0,40):
            temps = [t for t in (hourly.get(f"temperature_2m_member{i:02d}") or []) if t is not None]
            if temps: highs.append(max(temps))
        return highs
    except: return []

async def fetch_tomorrow_io(session, lat, lon, ds) -> float | None:
    if not TOMORROW_IO_KEY: return None
    try:
        async with session.get("https://api.tomorrow.io/v4/weather/forecast",
            params={"location":f"{lat},{lon}","apikey":TOMORROW_IO_KEY,"units":"imperial",
                    "timesteps":"1d","fields":"temperatureMax",
                    "startTime":f"{ds}T00:00:00Z","endTime":f"{ds}T23:59:59Z"}) as r:
            data = await r.json()
            days = data.get("timelines",{}).get("daily",[])
            return days[0].get("values",{}).get("temperatureMax") if days else None
    except: return None

async def get_forecast(session, semaphore, city_code, target_date) -> dict | None:
    info = CITY_COORDS.get(city_code)
    if not info: return None
    lat, lon    = info[0], info[1]
    ds          = target_date.isoformat()
    station     = WETHR_STATIONS.get(city_code, "")
    city_tz     = get_city_tz(city_code)
    local_hour  = datetime.now(city_tz).hour
    is_same_day = target_date == datetime.now(ET_TZ).date()

    # ── WETHR DETERMINISTIC MODELS (cache-only — prefetched in run_scan_async) ─
    wethr_models = {}
    if WETHR_API_KEY and station:
        wethr_models = fetch_wethr_all_models(station, ds, cache_only=True)

    # ── OPEN-METEO ENSEMBLE (for spread calculation) ──────────────────────────
    async with semaphore:
        (gfs, ecmwf_ens, icon_ens, nbm_prob, tomorrow) = await asyncio.gather(
            fetch_gfs_ensemble(session, lat, lon, ds),
            fetch_ecmwf_ensemble(session, lat, lon, ds),
            fetch_icon_ensemble(session, lat, lon, ds),
            fetch_nbm_probabilistic(session, lat, lon, ds),
            fetch_tomorrow_io(session, lat, lon, ds),
            return_exceptions=True,
        )
    if isinstance(gfs, Exception):       gfs       = []
    if isinstance(ecmwf_ens, Exception): ecmwf_ens = []
    if isinstance(icon_ens, Exception):  icon_ens  = []
    if isinstance(nbm_prob, Exception):  nbm_prob  = []
    if isinstance(tomorrow, Exception):  tomorrow  = None

    # Fallback to Open-Meteo deterministic if Wethr unavailable
    wethr_hrrr  = wethr_models.get("HRRR")
    wethr_nbm   = wethr_models.get("NBM")
    wethr_ecmwf = wethr_models.get("ECMWF-IFS")
    wethr_rap   = wethr_models.get("RAP")
    wethr_nam   = wethr_models.get("NAM4KM")
    wethr_gfs   = wethr_models.get("GFS")
    wethr_nws   = wethr_models.get("NWS")

    if not wethr_models:
        # Fall back to Open-Meteo deterministic
        async with semaphore:
            (ecmwf, hrrr, nbm, icon) = await asyncio.gather(
                fetch_ecmwf(session, lat, lon, ds),
                fetch_hrrr(session, lat, lon, ds),
                fetch_nbm(session, lat, lon, ds),
                fetch_icon(session, lat, lon, ds),
                return_exceptions=True,
            )
        if isinstance(ecmwf, Exception): ecmwf = None
        if isinstance(hrrr,  Exception): hrrr  = None
        if isinstance(nbm,   Exception): nbm   = None
        if isinstance(icon,  Exception): icon  = None
        wethr_ecmwf = ecmwf
        wethr_hrrr  = hrrr
        wethr_nbm   = nbm

    # Build available model list
    det_vals_wethr = {k: v for k, v in wethr_models.items() if v is not None}
    if tomorrow: det_vals_wethr["Tomorrow"] = tomorrow

    available = list(det_vals_wethr.keys())
    print(
        f"[forecast] {city_code} | wethr={len(wethr_models)} "
        f"({', '.join(wethr_models.keys()) or 'none'}) | "
        f"GFS={len(gfs)}mbrs ECMWF_ens={len(ecmwf_ens)}mbrs"
    )

    if len(available) < 2:
        print(f"[v3.26] {city_code} skipped: only {len(available)} models available (need 2)")
        return None

    # ── BUILD BLEND (v3.32: accuracy-driven auto-weights) ─────────────────────
    all_members = list(gfs) + list(ecmwf_ens) + list(icon_ens)

    # Pull per-model accuracy weights (cache-only during scan; prefetched).
    accuracy = _model_accuracy_cache.get(station, {}) if station else {}
    auto_w   = accuracy_to_weights(accuracy)

    # Default fixed weights (used when accuracy data missing for a model)
    fixed_w = {"NBM": 5, "NWS": 3, "ECMWF-IFS": 2, "HRRR": 2, "RAP": 2,
               "NAM4KM": 1, "GFS": 1}

    def w_for(model):
        if model in auto_w:
            return auto_w[model]
        return fixed_w.get(model, 1)

    blend = list(all_members)
    if wethr_nbm:   blend += [wethr_nbm]   * w_for("NBM")
    if nbm_prob:    blend += nbm_prob      * 3
    if wethr_nws:   blend += [wethr_nws]   * w_for("NWS")
    if wethr_ecmwf: blend += [wethr_ecmwf] * w_for("ECMWF-IFS")
    if wethr_hrrr:  blend += [wethr_hrrr]  * w_for("HRRR")
    if wethr_rap:   blend += [wethr_rap]   * w_for("RAP")
    if wethr_nam:   blend += [wethr_nam]   * w_for("NAM4KM")
    if wethr_gfs:   blend += [wethr_gfs]   * w_for("GFS")
    if tomorrow:    blend.append(tomorrow)
    if not blend:   return None

    if auto_w:
        wstr = " ".join(f"{m}:{w}" for m, w in sorted(auto_w.items(), key=lambda x:-x[1]))
        print(f"[autoweight] {city_code} {wstr}")

    mean = sum(blend) / len(blend)

    # ── SPREAD CALCULATION ────────────────────────────────────────────────────
    if len(all_members) >= 2:
        am     = sum(all_members) / len(all_members)
        spread = math.sqrt(sum((x-am)**2 for x in all_members) / len(all_members))
    elif len(blend) >= 2:
        spread = math.sqrt(sum((x-mean)**2 for x in blend) / len(blend))
    else:
        spread = 2.5

    det_list = [v for v in det_vals_wethr.values() if v is not None]
    if len(det_list) >= 2:
        det_range = max(det_list) - min(det_list)
        if det_range > 4.0:
            spread = max(spread, det_range / 2)
            print(f"[spread] {city_code} widened: det_range={det_range:.1f}F")

    if len(det_list) >= 4:
        det_range     = max(det_list) - min(det_list)
        dynamic_floor = 1.5 if det_range <= 2.0 else (2.0 if det_range <= 4.0 else det_range/2)
    elif len(det_list) >= 2:
        dynamic_floor = 2.5
    else:
        dynamic_floor = 3.5

    if len(all_members) < 10:
        spread = max(spread, dynamic_floor)

    # ── LONG-LEAD-TIME SPREAD WIDENING (v3.29) ────────────────────────────────
    # A forecast made far from the peak is inherently less certain. Overnight
    # (8pm-6am local) and next-day bets are 12-18+ hours from the high, so the
    # normal same-day spread floor produces false confidence (e.g. 97% NO at
    # 1am). Widen the spread floor for these to reflect real uncertainty.
    is_overnight_window = (local_hour >= 20 or local_hour < 6)
    if not is_same_day:
        # Next-day forecast — widest uncertainty
        spread = max(spread, 4.0)
        print(f"[spread] {city_code} next-day floor → ±{spread:.1f}F")
    elif is_overnight_window:
        # Same calendar day but pre-dawn — still 10-15h from peak
        spread = max(spread, 3.5)
        print(f"[spread] {city_code} overnight floor → ±{spread:.1f}F")

    # ── PACING CORRECTION (same-day only, after noon) ─────────────────────────
    pacing = {}
    if is_same_day and local_hour >= 12:
        with _lock:
            current_obs = asos_observed.get(city_code)
        if current_obs is not None:
            pacing = calculate_model_pacing(wethr_models, current_obs)
            mean   = apply_pacing_correction(mean, pacing, local_hour)

    # ── BIAS + FINAL ──────────────────────────────────────────────────────────
    bias           = get_bias(city_code)
    corrected_mean = mean + bias

    conf = "high" if spread < 2.0 else ("medium" if spread < 4.0 else "low")
    return {
        "ensemble_mean":    round(mean, 1),
        "corrected_mean":   round(corrected_mean, 1),
        "bias_applied":     round(bias, 2),
        "spread":           round(spread, 2),
        "ecmwf_high":       round(wethr_ecmwf, 1) if wethr_ecmwf else None,
        "hrrr_high":        round(wethr_hrrr, 1)  if wethr_hrrr  else None,
        "nbm_high":         round(wethr_nbm, 1)   if wethr_nbm   else None,
        "rap_high":         round(wethr_rap, 1)   if wethr_rap   else None,
        "icon_high":        None,
        "tomorrow_high":    round(tomorrow, 1)    if tomorrow    else None,
        "aifs_high":        None,
        "nws_high":         round(wethr_nws, 1)   if wethr_nws   else None,
        "nam4km_high":      round(wethr_nam, 1)   if wethr_nam   else None,
        "pacing":           pacing,
        "gfs_members":      len(gfs),
        "ecmwf_members":    len(ecmwf_ens),
        "icon_ens_members": len(icon_ens),
        "nbm_pct_points":   len(nbm_prob),
        "total_members":    len(all_members),
        "confidence":       conf,
        "nbm_weight":       5,
    }

# ── PROBABILITY MODEL ─────────────────────────────────────────────────────────
def _normal_cdf(x: float, mean: float, spread: float) -> float:
    if spread == 0:
        return 0.0 if x < mean else 1.0
    z = (x - mean) / spread
    t = 1 / (1 + 0.2316419 * abs(z))
    p = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    phi = 1 - (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * p
    return round(phi if z >= 0 else 1 - phi, 6)

def model_probability(forecast: dict, threshold: float, city_code: str,
                      kind: str = "T", lo: float = None, hi: float = None,
                      is_next_day: bool = False) -> float:
    mean   = forecast["corrected_mean"]
    spread = max(forecast["spread"], 0.5)

    with _lock:
        obs_high = asos_observed.get(city_code)

    asos_weight = 0.0
    asos_prob   = 0.5

    if obs_high is not None and not is_next_day:
        city_tz   = get_city_tz(city_code)
        now_local = datetime.now(city_tz)
        hour      = now_local.hour
        if hour >= 18:   asos_weight = 0.95
        elif hour >= 16: asos_weight = 0.85
        elif hour >= 14: asos_weight = 0.60
        elif hour >= 12: asos_weight = 0.40
        else:            asos_weight = 0.0

    if kind == "B" and lo is not None and hi is not None:
        ensemble_prob = _normal_cdf(hi, mean, spread) - _normal_cdf(lo, mean, spread)
        if obs_high is not None and not is_next_day:
            asos_prob = 0.97 if lo <= obs_high < hi else 0.03
    else:
        ensemble_prob = 1.0 - _normal_cdf(threshold, mean, spread)
        if obs_high is not None and not is_next_day:
            asos_prob = 0.98 if obs_high >= threshold else 0.02

    if obs_high is not None and not is_next_day:
        prob = asos_weight * asos_prob + (1 - asos_weight) * ensemble_prob
    else:
        prob = ensemble_prob

    return round(max(0.01, min(0.99, prob)), 4)

def longshot_probability_adjustment(implied_p: float) -> float:
    if implied_p < 0.10: return -0.03
    if implied_p > 0.90: return +0.03
    return 0.0

# ── ASOS YES CHECK ────────────────────────────────────────────────────────────
def asos_confirms_yes(city_code: str, kind: str, lo: float, hi: float,
                      threshold: float, forecast: dict, local_hour: int) -> bool:
    """
    v3.22: Uses heating profiles to project where the temp will peak.
    Only fires YES if projected peak lands inside the bucket.
    """
    if forecast["spread"] > MAX_SPREAD_YES:
        return False
    with _lock:
        obs_high = asos_observed.get(city_code)
    if obs_high is None:
        return False
    if kind != "B":
        return False

    # Project peak using heating profile
    projected_rise = get_projected_rise(city_code, local_hour)
    projected_peak = obs_high + projected_rise

    # YES wins if projected peak lands inside bucket (lo to hi)
    if lo <= projected_peak < hi:
        print(f"[v3.22] {city_code} ASOS YES: obs={obs_high}°F + rise={projected_rise:.1f}°F → projected={projected_peak:.1f}°F in bucket {lo}-{hi}")
        return True

    print(f"[v3.22] {city_code} ASOS YES blocked: obs={obs_high}°F + rise={projected_rise:.1f}°F → projected={projected_peak:.1f}°F NOT in bucket {lo}-{hi}")
    return False

# ── DISCORD ───────────────────────────────────────────────────────────────────
def post_discord(webhook, content, embeds=None):
    if not webhook: return
    try:
        requests.post(webhook, json={"content":content,"embeds":embeds or []}, timeout=10)
    except Exception as e:
        print(f"[discord] {e}")

# ── ORDERBOOK / LIQUIDITY (v3.30, fixed format v3.31) ─────────────────────────
def fetch_orderbook_liquidity(ticker: str, side: str) -> dict | None:
    """
    Fetch the Kalshi orderbook and compute, for the given side (YES or NO),
    the best take price, liquidity at that price, and total resting liquidity.

    Kalshi response format (orderbook_fp):
      { "orderbook_fp": {
          "yes_dollars": [["0.4200","13.00"], ...],   # YES bids, price+count strings
          "no_dollars":  [["0.5600","10.00"], ...] }}  # NO bids
    Arrays are ordered worst→best, so the BEST bid is the LAST element.

    To TAKE the YES side, you cross the best NO bid → pay (1.00 - best_no_bid).
    To TAKE the NO side,  you cross the best YES bid → pay (1.00 - best_yes_bid).
    """
    try:
        r = requests.get(f"{KALSHI_BASE}/markets/{ticker}/orderbook", timeout=8)
        r.raise_for_status()
        ob = r.json().get("orderbook_fp", {})

        yes_bids = ob.get("yes_dollars") or []
        no_bids  = ob.get("no_dollars")  or []

        # Opposing side we cross to take our position
        opposing = no_bids if side == "YES" else yes_bids
        if not opposing:
            return None

        # Parse to (price_dollars, count) floats; best bid = highest price
        parsed = []
        for entry in opposing:
            try:
                price = float(entry[0])
                count = float(entry[1])
                parsed.append((price, count))
            except (ValueError, IndexError, TypeError):
                continue
        if not parsed:
            return None

        parsed.sort(key=lambda x: -x[0])  # best (highest) bid first
        best_opp_price, best_opp_qty = parsed[0]

        take_price_cents = round((1.0 - best_opp_price) * 100)
        liq_at_best      = best_opp_qty
        total_liq        = sum(c for _, c in parsed)
        # Dollar value at best price = contracts × take_price
        dollars_at_best  = liq_at_best * (take_price_cents / 100)
        dollars_total    = total_liq   * (take_price_cents / 100)

        return {
            "best_price_cents": take_price_cents,
            "liq_at_best":      liq_at_best,
            "total_liq":        total_liq,
            "dollars_at_best":  dollars_at_best,
            "dollars_total":    dollars_total,
        }
    except Exception as e:
        print(f"[orderbook] {ticker}: {e}")
        return None

def format_liquidity_message(city: str, subtitle: str, side: str,
                             liq: dict | None) -> str:
    """Build the follow-up message text with price + liquidity."""
    if liq is None:
        return f"💧 **{city} — {subtitle}** | no resting {side} liquidity"
    bp  = liq["best_price_cents"]
    lab = liq["liq_at_best"]
    tot = liq["total_liq"]
    dab = liq["dollars_at_best"]
    dt  = liq["dollars_total"]
    return (
        f"💧 **{city} liquidity** ({side})\n"
        f"• Best price: **{bp}¢**\n"
        f"• At {bp}¢: **${dab:,.0f}** ({lab:,.0f} contracts)\n"
        f"• Total {side} side: **${dt:,.0f}** ({tot:,.0f} contracts)"
    )

def recommend_units(ev_pct, confidence, afd_hit, is_fire, is_next_day,
                    is_yes=False, pace_confirmed=False) -> float:
    if is_yes:
        base = 1.0 if ev_pct >= 30 else 0.5
        return max(base, 1.0) if pace_confirmed else base
    if is_next_day:
        if ev_pct >= 35:   base = 1.0 if confidence == "high" else 0.5
        elif ev_pct >= 25: base = 0.5 if confidence == "high" else 0.0
        else:              base = 0.0
        return base  # next-day can't be pace-confirmed (no live obs)
    if ev_pct >= 25:
        if confidence == "high":     base = 2.0
        elif confidence == "medium": base = 1.0
        else:                        base = 0.5
    elif ev_pct >= 15:
        if confidence == "high":     base = 1.0
        elif confidence == "medium": base = 0.5
        else:                        base = 0.0
    else:
        base = 0.0
    # Pace-confirmed bets are high-confidence by observation — floor at 1.5u
    # because the temperature physically can't reach the bucket, regardless
    # of what the EV says. Confidence, not just EV, should drive sizing here.
    if pace_confirmed:
        return max(base, 1.5)
    return base

def format_units(units: float) -> str:
    if units == 0: return "0u — flag only"
    return f"{int(units)}u" if units == int(units) else f"{units}u"

def build_embed(market, forecast, ev_data, obs_high, afd_hit,
                units=0, is_next_day=False, asos_yes=False,
                category="morning", pace_confirmed=False) -> dict:
    city_code = market["city_code"]
    best      = ev_data["best_side"]
    side_data = ev_data[best]
    conf      = forecast["confidence"]
    t_ev      = side_data["taker_ev"]
    m_ev      = side_data["maker_ev"]
    # Color by category
    color = CATEGORY_COLORS.get(category, 0x854f0b)
    if asos_yes: color = 0x0099ff  # blue for ASOS-confirmed YES

    if pace_confirmed:
        cat_label = "✅ PACE-CONFIRMED"
    else:
        cat_label = f"{CATEGORY_EMOJI.get(category,'')} {category.upper()}"

    fields = [
        {"name":"Type",  "value":cat_label, "inline":True},
        {"name":"Side",  "value":f"{'🌡️ ASOS ' if asos_yes else ''}{best}", "inline":True},
        {"name":"EV%",   "value":f"+{t_ev}%",         "inline":True},
        {"name":"Units", "value":format_units(units),  "inline":True},
    ]
    fields += [
        {"name":"Model prob",     "value":f"{side_data['prob']}%",           "inline":True},
        {"name":"Kalshi implied", "value":f"{side_data['implied']}%",        "inline":True},
        {"name":"Ensemble mean",  "value":f"{forecast['corrected_mean']}°F", "inline":True},
        {"name":"Spread",         "value":f"±{forecast['spread']}°F",        "inline":True},
        {"name":"Confidence",     "value":conf.capitalize(),                 "inline":True},
        {"name":"Half Kelly",     "value":f"{side_data['taker_hk']}%",       "inline":True},
        {"name":"Maker EV",       "value":f"+{m_ev}%",                       "inline":True},
    ]
    # ASOS high only meaningful for same-day daytime bets. Overnight bets carry
    # a stale prior-day reading, so suppress it to avoid confusion (v3.33).
    _city_hour = datetime.now(get_city_tz(city_code)).hour
    _is_overnight = (category == "overnight") or (_city_hour >= 20 or _city_hour < 6)
    if obs_high is not None and not is_next_day and not _is_overnight:
        local_hr = datetime.now(get_city_tz(city_code)).strftime("%H:%M")
        fields.append({"name":"ASOS high","value":f"{obs_high}°F @ {local_hr} local","inline":True})
    elif _is_overnight and not is_next_day:
        fields.append({"name":"ASOS high","value":"n/a (overnight — day not started)","inline":True})
    if is_next_day:
        fields.append({"name":"Market type","value":"📅 Next-day forecast","inline":True})
    if asos_yes:
        fields.append({"name":"Signal","value":"🌡️ ASOS-confirmed YES","inline":True})

    sources = []
    for key, label in [("ecmwf_high","ECMWF"),("hrrr_high","HRRR"),
                        ("nbm_high","NBM"),("rap_high","RAP"),
                        ("nam4km_high","NAM4KM"),("nws_high","NWS"),
                        ("tomorrow_high","Tomorrow")]:
        if forecast.get(key):
            sources.append(f"{label} {forecast[key]}°F")
    if afd_hit: sources.append("AFD signal")
    if sources:
        fields.append({"name":"\u200b","value":" | ".join(sources),"inline":False})

    return {"color":color,"fields":fields,
            "footer":{"text":f"{market['ticker']} | {datetime.now(ET_TZ).strftime('%H:%M ET')}"}}

# ── KALSHI MARKET DISCOVERY ───────────────────────────────────────────────────
def parse_threshold_from_ticker(ticker: str) -> tuple[float | None, str]:
    parts = ticker.rsplit("-", 1)
    if len(parts) < 2: return None, ""
    suffix = parts[1]
    if suffix.startswith("T"):
        try: return float(suffix[1:]), "T"
        except ValueError: return None, ""
    if suffix.startswith("B"):
        try: return float(suffix[1:]) + 0.5, "B"
        except ValueError: return None, ""
    return None, ""

def parse_market(m: dict) -> dict | None:
    ticker = m.get("ticker", "")
    if not ticker: return None
    event_ticker = m.get("event_ticker", "") or ""
    series = event_ticker.split("-")[0] if event_ticker else ""
    if not any(series.startswith(p) for p in ("KXHIGH","KXHIGHT","KXLOWT")): return None
    if series.startswith("KXLOWT"):    return None
    elif series.startswith("KXHIGHT"): city_code = series[len("KXHIGHT"):]
    else:                              city_code = series[len("KXHIGH"):]
    if not city_code: return None
    threshold, kind = parse_threshold_from_ticker(ticker)
    if threshold is None: return None
    yes_price = max(1, min(99, round(float(m.get("yes_ask_dollars") or 0.5) * 100)))
    no_price  = max(1, min(99, round(float(m.get("no_ask_dollars")  or 0.5) * 100)))
    subtitle  = m.get("title", "") or m.get("yes_sub_title", "") or ticker
    return {"ticker":ticker,"series":series,"city_code":city_code,
            "threshold_f":threshold,"threshold_kind":kind,
            "yes_price":yes_price,"no_price":no_price,"subtitle":subtitle}

ALL_TEMP_SERIES = [
    "KXHIGHNY","KXHIGHAUS","KXHIGHLAX","KXHIGHCHI","KXHIGHMIA",
    "KXHIGHTDAL","KXHIGHTDC","KXHIGHTSEA","KXHIGHTPHX","KXHIGHTBOS",
    "KXHIGHTHOU","KXHIGHTATL","KXHIGHTOKC","KXHIGHTLV","KXHIGHTSFO",
    "KXHIGHTDEN","KXHIGHTSA","KXHIGHTNO","KXHIGHTMN","KXHIGHTPHI",
    "KXHIGHTMEM","KXHIGHTPI","KXHIGHTBA","KXHIGHTCL","KXHIGHTSD",
    "KXHIGHTKC","KXHIGHTSL","KXHIGHTPO","KXHIGHTAL","KXHIGHTIN",
    "KXHIGHTEL","KXHIGHTMIL","KXHIGHTRAL","KXHIGHTTAM","KXHIGHTSLC",
    "KXHIGHTCOL","KXHIGHTTUC","KXHIGHTDE","KXHIGHTOL",
    "KXLOWTNYC","KXLOWTDAL","KXLOWTDC","KXLOWTSEA","KXLOWTPHX",
    "KXLOWTBOS","KXLOWTHOU","KXLOWTATL","KXLOWTOKC","KXLOWTLV",
    "KXLOWTSFO","KXLOWTAUS","KXLOWTLAX","KXLOWTCHI","KXLOWTMIA",
    "KXLOWTDEN","KXLOWTSA","KXLOWTNO","KXLOWTMN","KXLOWTPHI",
    "KXLOWTMEM","KXLOWTPI","KXLOWTBA","KXLOWTCL","KXLOWTSD",
    "KXLOWTKC","KXLOWTSL","KXLOWTPO","KXLOWTAL","KXLOWTIN",
    "KXLOWTEL","KXLOWTMIL","KXLOWTRAL","KXLOWTTAM","KXLOWTSLC",
    "KXLOWTCOL","KXLOWTTUC","KXLOWTDE","KXLOWTOL",
]
KXHIGH_SERIES = ALL_TEMP_SERIES

_market_cache: list[dict] = []
_market_cache_ts: float   = 0.0
MARKET_CACHE_TTL          = 280

def get_active_kalshi_markets() -> list[dict]:
    global _market_cache, _market_cache_ts
    with _lock:
        cache_age = time.time() - _market_cache_ts
        if _market_cache and cache_age < MARKET_CACHE_TTL:
            raw = list(_market_cache)
            markets = [m for m in (parse_market(r) for r in raw) if m]
            print(f"[kalshi] {len(markets)} markets (cache {cache_age:.0f}s old)")
            return markets
    markets_raw = []
    for series in KXHIGH_SERIES:
        try:
            r = requests.get(f"{KALSHI_BASE}/markets",
                params={"status":"open","series_ticker":series,"limit":25}, timeout=10)
            if r.status_code == 429:
                print(f"[kalshi] Rate limited, waiting 60s...")
                time.sleep(60)
                r = requests.get(f"{KALSHI_BASE}/markets",
                    params={"status":"open","series_ticker":series,"limit":25}, timeout=10)
                if r.status_code == 429:
                    print(f"[kalshi] Still rate limited, waiting 120s...")
                    time.sleep(120)
                    continue
            r.raise_for_status()
            markets_raw.extend(r.json().get("markets", []))
        except Exception as e:
            print(f"[kalshi] {series}: {e}")
        time.sleep(1.0)  # increased from 0.5 to 1.0 to avoid rate limits
    markets = [m for m in (parse_market(r) for r in markets_raw) if m]
    print(f"[kalshi] {len(markets)} active temperature markets")
    return markets

# ── TICKER DATE HELPER ────────────────────────────────────────────────────────
def ticker_date(ticker: str) -> date:
    try:
        part   = ticker.split("-")[1]
        months = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                  "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
        yr  = 2000 + int(part[0:2])
        mon = months[part[2:5]]
        day = int(part[5:7])
        return date(yr, mon, day)
    except:
        return date.today()

# ── ASYNC SCAN ────────────────────────────────────────────────────────────────
async def scan_market_async(session, semaphore, market, today, afd_cities):
    cc = market["city_code"]
    if cc not in CITY_COORDS: return None

    settle_date = ticker_date(market["ticker"])
    is_next_day = settle_date > today
    target_date = settle_date

    city_tz_check   = get_city_tz(cc)
    now_local_check = datetime.now(city_tz_check)

    # Alert key — anchored to SETTLEMENT date (target_date), not 'today', so
    # late-night bets for tomorrow's settlement don't get a fresh key after the
    # ET midnight rollover. Afternoon (2pm-8pm local) gets a separate re-alert
    # key; overnight and morning share one key per settlement date (v3.33).
    _ck_hour = now_local_check.hour
    if not is_next_day and 14 <= _ck_hour < 20:
        alert_key = f"{market['ticker']}_{target_date}_afternoon"
    else:
        alert_key = f"{market['ticker']}_{target_date}"

    with _lock:
        if alert_key in posted_alert_keys:
            return None

    # v3.21/v3.33: DB dedup — check if ticker already logged for this settlement
    # date. Anchored on market_date (settlement) so it survives ET midnight
    # rollover and restarts. Afternoon re-alerts allowed once per afternoon.
    if BIAS_LOGGING:
        try:
            import psycopg2
            conn = psycopg2.connect(os.environ.get("DATABASE_URL", ""))
            cur  = conn.cursor()
            if "afternoon" in alert_key:
                # Only re-alert if no afternoon-window entry exists for this
                # settlement date (compare against 2pm local ~ logged_at time).
                cur.execute(
                    "SELECT 1 FROM predictions WHERE ticker = %s AND market_date = %s "
                    "AND bet_category IN ('pacing','pace_confirmed') LIMIT 1",
                    (market["ticker"], str(target_date))
                )
            else:
                cur.execute(
                    "SELECT 1 FROM predictions WHERE ticker = %s AND market_date = %s LIMIT 1",
                    (market["ticker"], str(target_date))
                )
            already_logged = cur.fetchone() is not None
            conn.close()
            if already_logged:
                with _lock:
                    posted_alert_keys.add(alert_key)  # sync memory with DB
                return None
        except Exception as e:
            print(f"[v3.21] DB dedup check failed: {e} — using memory only")

    forecast = await get_forecast(session, semaphore, cc, target_date)
    if not forecast: return None

    threshold = market["threshold_f"]
    kind      = market["threshold_kind"]
    lo        = threshold - 0.5 if kind == "B" else None
    hi        = threshold + 0.5 if kind == "B" else None

    # ── ASOS-CONFIRMED YES (v3.19) ────────────────────────────────────────────
    # Only fires 2pm-5pm local, same-day B-type, ASOS within 1F of bucket
    asos_yes = False
    if (kind == "B" and not is_next_day and
            14 <= now_local_check.hour < 18 and
            asos_confirms_yes(cc, kind, lo, hi, threshold, forecast, now_local_check.hour)):
        # Check EV on YES side
        prob_yes = model_probability(forecast, threshold, cc, kind="B",
                                     lo=lo, hi=hi, is_next_day=False)
        ev_yes = compute_ev_kelly(prob_yes, market["yes_price"], market["no_price"])
        if (ev_yes["best_side"] == "YES" and
                ev_yes["YES"]["taker_ev"] >= YES_EV_THRESHOLD * 100):
            asos_yes = True
            prob     = prob_yes
            ev_data  = ev_yes
            best     = "YES"
            print(f"[v3.19] {cc} ASOS YES fired: EV={ev_yes['YES']['taker_ev']}%")

    if not asos_yes:
        # ── STANDARD NO PATH ─────────────────────────────────────────────────
        # Determine if this is a long-lead bet (overnight or next-day) which
        # needs a bigger gap since there's no observation to confirm it.
        _ov_hour = now_local_check.hour
        is_long_lead = is_next_day or (_ov_hour >= 20 or _ov_hour < 6)
        gap_multiplier = 1.5 if is_long_lead else 1.0

        if kind == "B":
            prob = model_probability(forecast, threshold, cc, kind="B",
                                     lo=lo, hi=hi, is_next_day=is_next_day)
            # v3.19: 1x spread same-day; v3.29: 1.5x for overnight/next-day
            if prob < 0.5:  # NO bet
                gap     = lo - forecast["corrected_mean"]
                min_gap = forecast["spread"] * gap_multiplier
                if gap < min_gap:
                    print(f"[v3.29] {cc} B-type NO blocked: gap={gap:.1f}F < min={min_gap:.1f}F "
                          f"(x{gap_multiplier} {'long-lead' if is_long_lead else 'same-day'})")
                    return None
        else:
            prob = model_probability(forecast, threshold, cc, kind="T",
                                     is_next_day=is_next_day)

        # Block T-type YES when mean >= threshold
        if kind == "T" and prob > 0.5 and forecast["corrected_mean"] >= threshold:
            print(f"[v3.16] {cc} T-type YES blocked: mean={forecast['corrected_mean']}°F >= thresh={threshold}°F")
            return None

        # Block next-day T-type YES
        if kind == "T" and is_next_day and prob > 0.5:
            return None

        # Block T-type NO when threshold too close to mean
        if kind == "T" and prob < 0.5:
            gap = threshold - forecast["corrected_mean"]
            if gap < forecast["spread"]:
                print(f"[v3.16] {cc} T-type NO blocked: gap={gap:.1f}F < spread={forecast['spread']}F")
                return None

        # Block T-type YES when ASOS near threshold after noon
        with _lock:
            obs_now = asos_observed.get(cc)
        if obs_now is not None and not is_next_day and kind == "T":
            if prob > 0.5 and now_local_check.hour >= 12 and obs_now >= threshold - 3.0:
                print(f"[v3.16] {cc} T-type YES blocked: ASOS={obs_now}°F near thresh={threshold}°F")
                return None

        implied_p = market["yes_price"] / 100
        adj       = longshot_probability_adjustment(implied_p)
        prob      = max(0.01, min(0.99, prob + adj))

        ev_data = compute_ev_kelly(prob, market["yes_price"], market["no_price"])
        best    = ev_data["best_side"]
        if not best: return None

        # Block all non-ASOS YES bets
        if best == "YES":
            return None

        # Block NO priced at 5 cents or less
        if best == "NO" and market["no_price"] <= 5:
            return None

    side_data = ev_data[best]
    t_ev      = side_data["taker_ev"]
    spread    = forecast["spread"]

    fire_thresh  = 0.40 if is_next_day else FIRE_EV_THRESHOLD
    watch_thresh = 0.30 if is_next_day else WATCH_EV_THRESHOLD

    if asos_yes:
        fire  = t_ev >= YES_EV_THRESHOLD * 100
        watch = fire
    else:
        fire  = t_ev >= fire_thresh * 100 and spread <= MAX_SPREAD_FIRE
        watch = t_ev >= watch_thresh * 100 and spread <= MAX_SPREAD_WATCH

    with _lock:
        obs_high = asos_observed.get(cc)
    afd_hit  = bool(afd_cities and cc in afd_cities)

    # ── PACE-CONFIRMED detection (v3.27, refined v3.28) ───────────────────────
    # A bet is "pace-confirmed" when it's a same-day afternoon NO bet and the
    # observed trajectory makes the outcome high-confidence. We project the peak
    # from current obs + remaining rise and check it lands well below the bucket.
    #
    # The required margin ADAPTS to today's conditions:
    #   - base margin = 1.5F
    #   - +extra if today is heating faster than normal (morning rate check)
    #   - +extra if the air is unusually dry (dewpoint depression)
    # This prevents over-confirming on abnormal days where the profile average
    # underestimates the peak.
    pace_confirmed = False
    if (not asos_yes and best == "NO" and kind == "B" and not is_next_day
            and 12 <= now_local_check.hour < 20 and obs_high is not None):
        projected_rise = get_projected_rise(cc, now_local_check.hour)
        projected_peak = obs_high + projected_rise

        # Adaptive margin requirement
        heat_flag    = get_heating_rate_flag(cc, now_local_check.hour, obs_high)
        dry_flag     = get_dryness_flag(cc)
        required_margin = 1.5 + heat_flag + dry_flag

        margin = lo - projected_peak
        if margin >= required_margin:
            pace_confirmed = True
            print(f"[v3.28] {cc} PACE-CONFIRMED NO: obs={obs_high}°F + rise={projected_rise:.1f}°F "
                  f"→ proj={projected_peak:.1f}°F, bucket_lo={lo}°F, margin={margin:.1f}°F "
                  f"(req={required_margin:.1f}F: base 1.5 +heat {heat_flag:+.1f} +dry {dry_flag:+.1f})")
        elif margin >= 1.5:
            # Would have confirmed under old rule but today's conditions say be cautious
            print(f"[v3.28] {cc} pace NOT confirmed: margin={margin:.1f}F < required={required_margin:.1f}F "
                  f"(heat {heat_flag:+.1f}, dry {dry_flag:+.1f}) — abnormal day, staying cautious")

    category = get_bet_category(cc, is_next_day, pace_confirmed)

    city    = CITY_COORDS[cc][2]
    day_tag = "tmrw" if is_next_day else "today"
    print(f"[scan] {city} [{kind}/{day_tag}/{best}/{category}]: model={side_data['prob']}% "
          f"implied={side_data['implied']}% EV={t_ev}% spread=±{spread}°F 🔥")

    if fire or watch or afd_hit:
        if BIAS_LOGGING:
            try:
                _log_prediction(
                    target_date,  # v3.19 fix: was 'today', now settlement date
                    cc, CITY_COORDS[cc][2], market["ticker"], threshold,
                    forecast, prob, market["yes_price"], market["no_price"],
                    obs_high, best_side=best,
                    taker_ev=ev_data[best]["taker_ev"], threshold_kind=kind,
                    bet_category=category,
                )
            except TypeError:
                # bias_logger may not yet support bet_category param
                try:
                    _log_prediction(
                        target_date, cc, CITY_COORDS[cc][2], market["ticker"], threshold,
                        forecast, prob, market["yes_price"], market["no_price"],
                        obs_high, best_side=best,
                        taker_ev=ev_data[best]["taker_ev"], threshold_kind=kind,
                    )
                except Exception as _e:
                    print(f"[bias_logger] {_e}")
            except Exception as _e:
                print(f"[bias_logger] {_e}")
        return {
            "market":         market,
            "forecast":       forecast,
            "ev_data":        ev_data,
            "obs_high":       obs_high,
            "afd":            afd_hit,
            "fire":           fire,
            "taker_ev":       t_ev,
            "threshold":      threshold,
            "kind":           kind,
            "alert_key":      alert_key,
            "is_next_day":    is_next_day,
            "asos_yes":       asos_yes,
            "category":       category,
            "pace_confirmed": pace_confirmed,
        }
    return None

# ── BUCKET DEDUPLICATION ──────────────────────────────────────────────────────
MAX_BETS_PER_CITY = 2  # v3.29: cap total bets per city per settlement date

def deduplicate_buckets(raw_results: list[dict]) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    for res in raw_results:
        key = (res["market"]["city_code"], res["kind"], res["is_next_day"])
        groups[key].append(res)

    kept = []
    for (cc, kind, next_day), alerts in groups.items():
        alerts.sort(key=lambda x: -x["taker_ev"])
        selected_thresholds = []
        for alert in alerts:
            thresh    = alert["threshold"]
            too_close = any(abs(thresh - s) < BUCKET_GAP_F for s in selected_thresholds)
            if not too_close:
                selected_thresholds.append(thresh)
                kept.append(alert)
                print(f"[dedup] {cc}/{kind}/{'tmrw' if next_day else 'today'} "
                      f"keeping thresh={thresh}°F EV={alert['taker_ev']}%")
            else:
                nearest = min(selected_thresholds, key=lambda s: abs(thresh - s))
                print(f"[dedup] {cc}/{kind} dropping thresh={thresh}°F (near {nearest}°F)")

    # ── PER-CITY CAP (v3.29) ──────────────────────────────────────────────────
    # Limit total bets per city per settlement date to avoid over-concentration.
    # Keep the highest-EV bets; pace-confirmed bets get priority (they're the
    # highest-confidence). Prevents the "3 OKC NO bets all miss" scenario.
    by_city_date: dict[tuple, list] = defaultdict(list)
    for alert in kept:
        ck = (alert["market"]["city_code"], alert["is_next_day"])
        by_city_date[ck].append(alert)

    final = []
    for (cc, next_day), alerts in by_city_date.items():
        if len(alerts) <= MAX_BETS_PER_CITY:
            final.extend(alerts)
        else:
            # Sort: pace-confirmed first, then by EV
            alerts.sort(key=lambda x: (not x.get("pace_confirmed", False),
                                       -x["taker_ev"]))
            keep = alerts[:MAX_BETS_PER_CITY]
            drop = alerts[MAX_BETS_PER_CITY:]
            final.extend(keep)
            for d in drop:
                print(f"[citycap] {cc}/{'tmrw' if next_day else 'today'} "
                      f"dropping thresh={d['threshold']}°F EV={d['taker_ev']}% "
                      f"(cap {MAX_BETS_PER_CITY}/city)")
    return final

async def run_scan_async(force_codes=None):
    maybe_reset_daily()
    ts      = datetime.now(ET_TZ).strftime("%H:%M ET")
    markets = get_active_kalshi_markets()
    if not markets:
        post_discord(DISCORD_LOG_WEBHOOK, f"📊 **Scan done** {ts} | 0 markets found")
        return
    if force_codes:
        markets = [m for m in markets if m["city_code"] in force_codes]

    today     = datetime.now(ET_TZ).date()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT+4)
    timeout   = aiohttp.ClientTimeout(total=30)

    with _lock:
        afd_cities = set(afd_flagged_cities)

    # ── PRE-FETCH WETHR FORECASTS (sequential, into cache) ────────────────────
    # This prevents 40 concurrent scan tasks from each hammering the Wethr API.
    # By loading everything into the cache first (sequentially, rate-limited),
    # the concurrent scan tasks below just read from cache = zero Wethr calls.
    if WETHR_API_KEY:
        cities_to_fetch = set()
        for m in markets:
            cc = m["city_code"]
            if cc in WETHR_STATIONS:
                cities_to_fetch.add(cc)

        # Determine which dates we need (today + any next-day markets)
        dates_needed = set()
        for m in markets:
            dates_needed.add(ticker_date(m["ticker"]).isoformat())

        prefetch_count = 0
        for cc in cities_to_fetch:
            station = WETHR_STATIONS.get(cc)
            if not station:
                continue
            # Pre-fetch model accuracy for this station (cached 6h)
            await asyncio.get_event_loop().run_in_executor(
                None, fetch_model_accuracy, station)
            for ds in dates_needed:
                # Check if already cached and fresh
                sample_key = f"{station}_HRRR_{ds}"
                now = time.time()
                if sample_key in _wethr_fcst_cache and \
                   now - _wethr_fcst_ts.get(sample_key, 0) < WETHR_FCST_TTL:
                    continue  # already cached, skip
                # Fetch all models for this city/date into cache (blocking)
                await asyncio.get_event_loop().run_in_executor(
                    None, fetch_wethr_all_models, station, ds)
                prefetch_count += 1

        if prefetch_count > 0:
            print(f"[prefetch] Loaded Wethr forecasts for {prefetch_count} city/date combos into cache")

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        raw_results = await asyncio.gather(
            *[scan_market_async(session,semaphore,m,today,afd_cities) for m in markets],
            return_exceptions=True,
        )

    valid_results = [r for r in raw_results if r is not None and not isinstance(r, Exception)]
    filtered      = deduplicate_buckets(valid_results)

    same_day_count = sum(1 for r in filtered if not r["is_next_day"])
    next_day_count = sum(1 for r in filtered if r["is_next_day"])
    yes_count      = sum(1 for r in filtered if r.get("asos_yes"))
    print(f"[dedup] {len(valid_results)} raw → {len(filtered)} after dedup "
          f"({same_day_count} same-day, {next_day_count} next-day, {yes_count} ASOS YES)")

    alerts = 0
    cat_counts = {"overnight": 0, "morning": 0, "pacing": 0, "pace_confirmed": 0}
    for res in filtered:
        market         = res["market"]
        forecast       = res["forecast"]
        ev_data        = res["ev_data"]
        is_next_day    = res["is_next_day"]
        asos_yes       = res.get("asos_yes", False)
        category       = res.get("category", "morning")
        pace_confirmed = res.get("pace_confirmed", False)
        city           = CITY_COORDS.get(market["city_code"],(None,None,market["city_code"]))[2]
        best           = ev_data["best_side"]
        t_ev_res       = ev_data[best]["taker_ev"]
        fire           = res["fire"]

        cat_counts[category] = cat_counts.get(category, 0) + 1

        # Emoji by category (ASOS YES overrides)
        if asos_yes:
            emoji = "🌡️"
        else:
            emoji = CATEGORY_EMOJI.get(category, "🔥")

        day_label = "📅 TOMORROW" if is_next_day else "📍 TODAY"
        units     = recommend_units(t_ev_res, forecast["confidence"],
                                    res["afd"], fire, is_next_day, asos_yes,
                                    pace_confirmed)
        embed     = build_embed(market, forecast, ev_data, res["obs_high"],
                                res["afd"], units, is_next_day, asos_yes,
                                category, pace_confirmed)
        post_discord(DISCORD_WEBHOOK_URL,
                     f"{emoji} **{city} — {market['subtitle']}** {day_label}", [embed])

        # v3.30: follow-up message with live price + liquidity from orderbook
        try:
            liq = fetch_orderbook_liquidity(market["ticker"], best)
            liq_msg = format_liquidity_message(city, market["subtitle"], best, liq)
            post_discord(DISCORD_WEBHOOK_URL, liq_msg)
        except Exception as _e:
            print(f"[orderbook] follow-up failed for {market['ticker']}: {_e}")

        with _lock:
            posted_alert_keys.add(res["alert_key"])
        alerts += 1

    msg = (f"📊 **Scan done** {ts} | {len(markets)} markets | "
           f"{len(valid_results)} raw | {alerts} posted "
           f"(🌙{cat_counts['overnight']} 🌅{cat_counts['morning']} 📈{cat_counts['pacing']} ✅{cat_counts['pace_confirmed']})")
    if force_codes: msg += f" | triggered: {', '.join(force_codes)}"
    post_discord(DISCORD_LOG_WEBHOOK, msg)
    print(f"[scan] Done — {alerts} posted")

def run_scan(force_codes=None):
    asyncio.run(run_scan_async(force_codes))

# ── PRICE WATCHER ─────────────────────────────────────────────────────────────
PRICE_POLL_SECS    = 120
PRICE_MOVE_TRIGGER = 3
price_snapshot: dict[str, tuple[int, int]] = {}

def fetch_current_prices() -> dict[str, tuple[int, int, str]]:
    global _market_cache, _market_cache_ts
    prices = {}
    markets_raw = []
    for series in KXHIGH_SERIES:
        try:
            r = requests.get(f"{KALSHI_BASE}/markets",
                params={"status":"open","series_ticker":series,"limit":25}, timeout=8)
            if r.status_code == 429:
                time.sleep(60)
                r = requests.get(f"{KALSHI_BASE}/markets",
                    params={"status":"open","series_ticker":series,"limit":25}, timeout=8)
                if r.status_code == 429:
                    continue
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[price_watcher] {series}: {e}")
            continue
        for m in data.get("markets", []):
            ticker    = m.get("ticker", "")
            city_code = series.replace("KXHIGH", "")
            yes_price = max(1, min(99, round(float(m.get("yes_ask_dollars") or 0.5) * 100)))
            no_price  = max(1, min(99, round(float(m.get("no_ask_dollars")  or 0.5) * 100)))
            if ticker:
                prices[ticker] = (yes_price, no_price, city_code)
                markets_raw.append(m)
        time.sleep(1.0)  # increased to avoid rate limits
    with _lock:
        _market_cache    = markets_raw
        _market_cache_ts = time.time()
    return prices

def price_watcher_loop():
    global price_snapshot
    print(f"[price_watcher] Starting — {PRICE_POLL_SECS}s poll, {PRICE_MOVE_TRIGGER}¢ trigger")
    price_snapshot = {t: (y, n) for t, (y, n, _) in fetch_current_prices().items()}
    time.sleep(PRICE_POLL_SECS)
    while True:
        current      = fetch_current_prices()
        moved_cities = set()
        for ticker, (yes_new, no_new, city_code) in current.items():
            prev = price_snapshot.get(ticker)
            if prev is None:
                price_snapshot[ticker] = (yes_new, no_new)
                continue
            yes_old, no_old = prev
            if abs(yes_new-yes_old) >= PRICE_MOVE_TRIGGER or abs(no_new-no_old) >= PRICE_MOVE_TRIGGER:
                moved_cities.add(city_code)
                direction = "↑" if yes_new > yes_old else "↓"
                print(f"[price_watcher] {city_code} {ticker}: {yes_old}¢→{yes_new}¢ {direction}")
            price_snapshot[ticker] = (yes_new, no_new)
        if moved_cities:
            known = {c for c in moved_cities if c in CITY_COORDS}
            if known:
                print(f"[price_watcher] Move detected → rescan: {known}")
                run_scan(force_codes=known)
        time.sleep(PRICE_POLL_SECS)

def signal_rescan_loop():
    while True:
        time.sleep(60)
        with _lock:
            cities = set(afd_flagged_cities)
            if cities:
                afd_flagged_cities.clear()
        if cities:
            print(f"[rescan] Signal-triggered: {cities}")
            run_scan(force_codes=cities)

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    print("🌡️  Kalshi Weather Bot v3.34")
    print(f"   v3.34: Cross-scan duplicate fix (alert key on settlement date)")
    print(f"          (v3.33: overnight ASOS fix; v3.32: Dev tier features)")
    print(f"          (v3.31: liquidity fix + pace_confirmed category)")
    print(f"   Cities: {len(CITY_COORDS)} | "
          f"WFOs: {len(set(info[4] for info in CITY_COORDS.values()))}")

    if not DISCORD_WEBHOOK_URL: print("[warn] DISCORD_WEBHOOK_URL not set")
    if not ANTHROPIC_API_KEY:   print("[warn] ANTHROPIC_API_KEY not set")
    if not WETHR_API_KEY:       print("[warn] WETHR_API_KEY not set — using Aviation Weather fallback")

    threading.Thread(target=asos_poll_loop,     daemon=True).start()
    threading.Thread(target=afd_scanner_loop,   daemon=True).start()
    threading.Thread(target=price_watcher_loop, daemon=True).start()
    threading.Thread(target=signal_rescan_loop, daemon=True).start()
    threading.Thread(target=wethr_push_loop,    daemon=True).start()

    print("[main] Waiting 100s for price watcher to populate market cache...")
    time.sleep(100)

    while True:
        try:
            now_et = datetime.now(ET_TZ)
            if now_et.weekday() == 3 and 3 <= now_et.hour < 5:
                print(f"[main] Kalshi maintenance window — sleeping 30 min")
                time.sleep(1800)
                continue
            run_scan()
        except Exception as e:
            print(f"[error] {e}")
        time.sleep(SCAN_INTERVAL_SECS)

if __name__ == "__main__":
    main()
