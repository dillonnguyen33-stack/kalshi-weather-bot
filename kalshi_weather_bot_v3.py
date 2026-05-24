"""
Kalshi Weather Temperature Bot — v3.4

Changes from v3.1:
  v3.2 — CDF bucket probability fix: B-type (bucket) markets now use
          CDF(hi)-CDF(lo) so all buckets sum to ~100%. Eliminates the
          impossible multi-bucket alerts (200 fake 90%+ EV signals).
  v3.3 — bias_logger fix: log_prediction() now called AFTER ev_data is
          computed, passing best_side, taker_ev, and threshold_kind so
          the calibration report has complete data.

Install: pip install aiohttp requests
"""

import os, asyncio, aiohttp, math, json, time, threading, requests, re

try:
    from bias_logger import log_prediction as _log_prediction
    BIAS_LOGGING = True
except ImportError:
    BIAS_LOGGING = False
from datetime import datetime, date, timezone, timedelta
from zoneinfo import ZoneInfo

# ── CONFIG ────────────────────────────────────────────────────────────────────
KALSHI_BASE         = "https://external-api.kalshi.com/trade-api/v2"
OPEN_METEO_BASE     = "https://api.open-meteo.com/v1"
OPEN_METEO_ENS_BASE = "https://ensemble-api.open-meteo.com/v1"
NWS_API_BASE        = "https://api.weather.gov"
AWC_METAR_BASE      = "https://aviationweather.gov/api/data/metar"

X_BEARER_TOKEN      = os.environ.get("X_BEARER_TOKEN", "")
ANTHROPIC_API_KEY   = os.environ.get("ANTHROPIC_API_KEY", "")
TOMORROW_IO_KEY     = os.environ.get("TOMORROW_IO_KEY", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
DISCORD_LOG_WEBHOOK = os.environ.get("DISCORD_LOG_WEBHOOK", "")

SCAN_INTERVAL_SECS  = 300
TWEET_POLL_SECS     = 600
ASOS_POLL_SECS      = 600
AFD_POLL_SECS       = 3600
FIRE_EV_THRESHOLD   = 0.25
WATCH_EV_THRESHOLD  = 0.15
MAX_SPREAD_FIRE     = 3.0
MAX_SPREAD_WATCH    = 5.0
MAX_CONCURRENT      = 8

ET_TZ  = ZoneInfo("America/New_York")
NWS_UA = "KalshiWeatherBot/3.3 dillonnguyen33@gmail.com"

# ── CITY CONFIG ───────────────────────────────────────────────────────────────
CITY_COORDS = {
    "NY":  (40.7128,  -74.0060, "New York City",    "KNYC", "OKX"),
    "NYC": (40.7128,  -74.0060, "New York City",    "KNYC", "OKX"),
    "AUS": (30.2672,  -97.7431, "Austin",           "KAUS", "EWX"),
    "LAX": (34.0522, -118.2437, "Los Angeles",      "KLAX", "LOX"),
    "CHI": (41.8781,  -87.6298, "Chicago",          "KMDW", "LOT"),
    "MIA": (25.7617,  -80.1918, "Miami",            "KMIA", "MFL"),
    "DAL": (32.7767,  -96.7970, "Dallas",           "KDFW", "FWD"),
    "DC":  (38.9072,  -77.0369, "Washington DC",    "KDCA", "LWX"),
    "SEA": (47.6062, -122.3321, "Seattle",          "KSEA", "SEW"),
    "PHX": (33.4484, -112.0740, "Phoenix",          "KPHX", "PSR"),
    "BOS": (42.3601,  -71.0589, "Boston",           "KBOS", "BOX"),
    "HOU": (29.7604,  -95.3698, "Houston",          "KIAH", "HGX"),
    "ATL": (33.7490,  -84.3880, "Atlanta",          "KATL", "FFC"),
    "OKC": (35.4676,  -97.5164, "Oklahoma City",    "KOKC", "OUN"),
    "LV":  (36.1699, -115.1398, "Las Vegas",        "KLAS", "VEF"),
    "SFO": (37.7749, -122.4194, "San Francisco",    "KSFO", "MTR"),
    "DEN": (39.7392, -104.9903, "Denver",           "KDEN", "BOU"),
    "SA":  (29.4241,  -98.4936, "San Antonio",      "KSAT", "EWX"),
    "NO":  (29.9511,  -90.0715, "New Orleans",      "KMSY", "LIX"),
    "MN":  (44.9778,  -93.2650, "Minneapolis",      "KMSP", "MPX"),
    "PHI": (39.9526,  -75.1652, "Philadelphia",     "KPHL", "PHI"),
    "MEM": (35.1495,  -90.0490, "Memphis",          "KMEM", "MEG"),
    "PI":  (40.4406,  -79.9959, "Pittsburgh",       "KPIT", "PBZ"),
    "BA":  (39.2904,  -76.6122, "Baltimore",        "KBWI", "LWX"),
    "CL":  (41.4993,  -81.6944, "Cleveland",        "KCLE", "CLE"),
    "SD":  (32.7157, -117.1611, "San Diego",        "KSAN", "SGX"),
    "KC":  (39.0997,  -94.5786, "Kansas City",      "KMCI", "EAX"),
    "SL":  (38.6270,  -90.1994, "St. Louis",        "KSTL", "LSX"),
    "PO":  (45.5051, -122.6750, "Portland",         "KPDX", "PQR"),
    "AL":  (35.2220,  -80.8431, "Charlotte",        "KCLT", "GSP"),
    "IN":  (39.7684,  -86.1581, "Indianapolis",     "KIND", "IND"),
    "COL": (39.9612,  -82.9988, "Columbus",         "KCMH", "ILN"),
    "TUC": (32.2226, -110.9747, "Tucson",           "KTUS", "TWC"),
    "EL":  (31.7619, -106.4850, "El Paso",          "KELP", "EPZ"),
    "MIL": (43.0389,  -87.9065, "Milwaukee",        "KMKE", "MKX"),
    "RAL": (35.7796,  -78.6382, "Raleigh",          "KRDU", "RAH"),
    "TAM": (27.9506,  -82.4572, "Tampa",            "KTPA", "TBW"),
    "SLC": (40.7608, -111.8910, "Salt Lake City",   "KSLC", "SLC"),
    "OL":  (36.1627,  -86.7816, "Nashville",        "KBNA", "OHX"),
    "DE":  (42.3314,  -83.0458, "Detroit",          "KDTW", "DTX"),
}

# ── BIAS CORRECTIONS ──────────────────────────────────────────────────────────
CITY_BIAS_F = {
    "NY":  [ 0.8,  0.7,  0.5,  0.3,  0.2,  0.0, -0.3, -0.2,  0.0,  0.3,  0.5,  0.7],
    "CHI": [ 1.2,  1.0,  0.8,  0.4,  0.2,  0.0, -0.5, -0.4,  0.0,  0.5,  0.8,  1.1],
    "LAX": [-0.5, -0.4, -0.3, -0.2, -0.2, -0.3, -0.4, -0.4, -0.3, -0.2, -0.3, -0.4],
    "MIA": [ 0.3,  0.3,  0.2,  0.1,  0.0,  0.0, -0.2, -0.2,  0.0,  0.1,  0.2,  0.3],
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

# ── RUNTIME STATE ─────────────────────────────────────────────────────────────
asos_observed: dict[str, float] = {}
afd_last_hash: dict[str, str]   = {}
seen_tweet_ids: set      = set()
seen_afd_ids:   set      = set()
tweet_flagged_cities: set = set()
afd_flagged_cities:   set = set()
_lock = threading.Lock()

# ── TWITTER ACCOUNTS ──────────────────────────────────────────────────────────
NWS_CITY_OFFICES = [
    "NWSNewYork","NWSPhiladelphia","NWSBaltimore","NWSChicago","NWSDetroit",
    "NWSCleveland","NWSPittsburgh","NWSLosAngeles","NWSSanDiego","NWSBayArea",
    "NWSMiami","NWSTampaBay","NWSJacksonville","NWSAtlanta","NWSCharlotte",
    "NWSRaleigh","NWSNashville","NWSMemphis","NWSNewOrleans","NWSHouston",
    "NWSSanAntonio","NWSDallas","NWSMinneapolis","NWSMilwaukee","NWSIndianapolis",
    "NWSColumbus","NWSCincinnati","NWSStLouis","NWSKansasCity","NWSOklahoma",
    "NWSDenver","NWSSaltLake","NWSPhoenix","NWSTucson","NWSLasVegas",
    "NWSSeattle","NWSPortland","NWSAlbuquerque","NWSElPaso","NWSBoston","NWSAlbany",
]
NWS_NATIONAL = [
    "NWS","NWSWPC","NWSCPC","NWStornado","NWSSevereTstorm",
    "NHC_Atlantic","NHC_Pacific","NWSstormreports",
]
MET_ACCOUNTS = [
    "JimCantore","StuOstro","Ariweather","mikebettes","chadmyersCNN",
    "capitalweather","AndrewFreedman","accuweather","MarkNegriBWX",
    "EricFisher","ReedTimmerAccu","ryanhallyall","spaghettimodels",
    "weatherbell","TaraWallace_Wx",
]
ALL_ACCOUNTS = NWS_CITY_OFFICES + NWS_NATIONAL + MET_ACCOUNTS

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
        p     = price / 100
        t_fee = kalshi_taker_fee(price)
        m_fee = kalshi_maker_fee(price)
        win   = 1 - p
        t_ev  = prob * (win - t_fee) - (1 - prob) * p
        m_ev  = prob * (win - m_fee) - (1 - prob) * p
        t_kelly = max(0, (prob * win - (1 - prob) * p) / win) if win > 0 else 0
        m_kelly = max(0, t_kelly * (1 + (t_fee - m_fee) / win)) if win > 0 else 0
        results[side] = {
            "prob":     round(prob * 100, 1),
            "implied":  round(p * 100, 1),
            "taker_ev": round(t_ev * 100, 1),
            "maker_ev": round(m_ev * 100, 1),
            "taker_hk": round(t_kelly * 0.5 * 100, 1),
            "maker_hk": round(m_kelly * 0.5 * 100, 1),
            "taker_fee_pct": round(t_fee / win * 100, 1) if win > 0 else 0,
        }
    best = max(("YES", "NO"), key=lambda s: results[s]["taker_ev"])
    if results[best]["taker_ev"] <= 0:
        best = None
    return {"best_side": best, "YES": results["YES"], "NO": results["NO"]}

# ── ASOS ──────────────────────────────────────────────────────────────────────
def fetch_asos_high(city_code: str) -> float | None:
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
        print(f"[asos] {city_code}/{icao}: {e}")
        return None

def asos_poll_loop():
    print(f"[asos] Starting observation poll for {len(CITY_COORDS)} cities")
    while True:
        for code in CITY_COORDS:
            high = fetch_asos_high(code)
            if high is not None:
                with _lock:
                    asos_observed[code] = high
        time.sleep(ASOS_POLL_SECS)

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
            headers={"x-api-key":ANTHROPIC_API_KEY,"anthropic-version":"2023-06-01","content-type":"application/json"},
            json={"model":"claude-sonnet-4-20250514","max_tokens":200,"system":system,
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
    while True:
        for wfo in wfos:
            text = fetch_afd_text(wfo)
            if not text: continue
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

async def fetch_hrrr(session, lat, lon, ds) -> float | None:
    try:
        data = await _get_json(session, f"{OPEN_METEO_BASE}/gfs", {
            "latitude":lat,"longitude":lon,"hourly":"temperature_2m",
            "temperature_unit":"fahrenheit","start_date":ds,"end_date":ds,"models":"hrrr"})
        temps = [t for t in (data.get("hourly",{}).get("temperature_2m") or []) if t is not None]
        return max(temps) if temps else None
    except: return None

async def fetch_rap(session, lat, lon, ds) -> float | None:
    try:
        data = await _get_json(session, f"{OPEN_METEO_BASE}/gfs", {
            "latitude":lat,"longitude":lon,"hourly":"temperature_2m",
            "temperature_unit":"fahrenheit","start_date":ds,"end_date":ds,"models":"rap"})
        temps = [t for t in (data.get("hourly",{}).get("temperature_2m") or []) if t is not None]
        return max(temps) if temps else None
    except: return None

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
        data = await _get_json(session, f"{OPEN_METEO_BASE}/gfs", {
            "latitude":lat,"longitude":lon,"hourly":"temperature_2m",
            "temperature_unit":"fahrenheit","start_date":ds,"end_date":ds,"models":"nbm"})
        temps = [t for t in (data.get("hourly",{}).get("temperature_2m") or []) if t is not None]
        return max(temps) if temps else None
    except: return None

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
    lat, lon = info[0], info[1]
    ds = target_date.isoformat()
    async with semaphore:
        ecmwf, hrrr, rap, gfs, nbm, icon, ecmwf_ens, tomorrow = await asyncio.gather(
            fetch_ecmwf(session, lat, lon, ds),
            fetch_hrrr(session, lat, lon, ds),
            fetch_rap(session, lat, lon, ds),
            fetch_gfs_ensemble(session, lat, lon, ds),
            fetch_nbm(session, lat, lon, ds),
            fetch_icon(session, lat, lon, ds),
            fetch_ecmwf_ensemble(session, lat, lon, ds),
            fetch_tomorrow_io(session, lat, lon, ds),
            return_exceptions=True,
        )
    if isinstance(ecmwf, Exception):     ecmwf     = None
    if isinstance(hrrr, Exception):      hrrr      = None
    if isinstance(rap, Exception):       rap       = None
    if isinstance(gfs, Exception):       gfs       = []
    if isinstance(nbm, Exception):       nbm       = None
    if isinstance(icon, Exception):      icon      = None
    if isinstance(ecmwf_ens, Exception): ecmwf_ens = []
    if isinstance(tomorrow, Exception):  tomorrow  = None

    all_members = list(gfs) + list(ecmwf_ens)
    if not all_members and ecmwf is None and hrrr is None and nbm is None:
        return None

    blend = list(all_members)
    if nbm:      blend += [nbm, nbm, nbm]
    if ecmwf:    blend += [ecmwf, ecmwf]
    if hrrr:     blend += [hrrr, hrrr]
    if rap:      blend += [rap, rap]
    if icon:     blend.append(icon)
    if tomorrow: blend.append(tomorrow)
    if not blend: return None

    mean = sum(blend) / len(blend)
    if len(all_members) >= 2:
        am = sum(all_members) / len(all_members)
        spread = math.sqrt(sum((x-am)**2 for x in all_members) / len(all_members))
    elif len(blend) >= 2:
        spread = math.sqrt(sum((x-mean)**2 for x in blend) / len(blend))
    else:
        spread = 2.5

    bias = get_bias(city_code)
    corrected_mean = mean + bias
    # Floor spread at 1.5°F when fewer than 10 ensemble members
    if len(all_members) < 10:
        spread = max(spread, 1.5)
    conf = "high" if spread < 2.0 else ("medium" if spread < 4.0 else "low")
    return {
        "ensemble_mean":  round(mean, 1),
        "corrected_mean": round(corrected_mean, 1),
        "bias_applied":   round(bias, 2),
        "spread":         round(spread, 2),
        "ecmwf_high":     round(ecmwf, 1)    if ecmwf    else None,
        "hrrr_high":      round(hrrr, 1)     if hrrr     else None,
        "nbm_high":       round(nbm, 1)      if nbm      else None,
        "rap_high":       round(rap, 1)      if rap      else None,
        "icon_high":      round(icon, 1)     if icon     else None,
        "tomorrow_high":  round(tomorrow, 1) if tomorrow else None,
        "gfs_members":    len(gfs),
        "ecmwf_members":  len(ecmwf_ens),
        "total_members":  len(all_members),
        "confidence":     conf,
    }

# ── PROBABILITY MODEL (v3.2: CDF-based, buckets sum to 100%) ─────────────────
def _normal_cdf(x: float, mean: float, spread: float) -> float:
    """P(X <= x) for a normal distribution."""
    if spread == 0:
        return 0.0 if x < mean else 1.0
    z = (x - mean) / spread
    t = 1 / (1 + 0.2316419 * abs(z))
    p = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    phi = 1 - (1 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * p
    return round(phi if z >= 0 else 1 - phi, 6)

def model_probability(forecast: dict, threshold: float, city_code: str,
                      kind: str = "T", lo: float = None, hi: float = None) -> float:
    """
    v3.2: TRUE probability using normal CDF fitted to ensemble mean+spread.
    B-type (bucket) markets: P(lo <= high < hi) = CDF(hi) - CDF(lo).
    T-type markets: P(high > threshold) = 1 - CDF(threshold).
    All buckets for a city now sum to ~100%.
    """
    mean   = forecast["corrected_mean"]
    spread = max(forecast["spread"], 0.5)

    with _lock:
        obs_high = asos_observed.get(city_code)

    asos_weight = 0.0
    asos_prob   = 0.5
    if obs_high is not None:
        now_et = datetime.now(ET_TZ)
        hour   = now_et.hour
        if hour >= 18:   asos_weight = 0.95
        elif hour >= 16: asos_weight = 0.85
        elif hour >= 14: asos_weight = 0.60
        elif hour >= 12: asos_weight = 0.40
        else:            asos_weight = 0.15

    if kind == "B" and lo is not None and hi is not None:
        ensemble_prob = _normal_cdf(hi, mean, spread) - _normal_cdf(lo, mean, spread)
        if obs_high is not None:
            asos_prob = 0.97 if lo <= obs_high < hi else 0.03
    else:
        ensemble_prob = 1.0 - _normal_cdf(threshold, mean, spread)
        if obs_high is not None:
            asos_prob = 0.98 if obs_high >= threshold else 0.02

    if obs_high is not None:
        prob = asos_weight * asos_prob + (1 - asos_weight) * ensemble_prob
    else:
        prob = ensemble_prob

    return round(max(0.01, min(0.99, prob)), 4)

def longshot_probability_adjustment(implied_p: float) -> float:
    if implied_p < 0.10: return -0.03
    if implied_p > 0.90: return +0.03
    return 0.0

# ── DISCORD ───────────────────────────────────────────────────────────────────
def post_discord(webhook, content, embeds=None):
    if not webhook: return
    try:
        requests.post(webhook, json={"content":content,"embeds":embeds or []}, timeout=10)
    except Exception as e:
        print(f"[discord] {e}")

def recommend_units(ev_pct, confidence, tweet_hit, afd_hit, is_fire) -> float:
    signal_boost = tweet_hit and afd_hit
    if ev_pct >= 25:
        if confidence == "high":   return 2.0 if signal_boost else 1.5
        elif confidence == "medium": return 1.0
        else: return 0.5
    elif ev_pct >= 15:
        if confidence == "high":   return 1.0
        elif confidence == "medium": return 0.5
        else: return 0.0
    return 0.0

def format_units(units: float) -> str:
    if units == 0: return "0u — flag only"
    return f"{int(units)}u" if units == int(units) else f"{units}u"

def build_embed(market, forecast, ev_data, obs_high, tweet_hit, afd_hit, units=0) -> dict:
    city_code = market["city_code"]
    city_name = CITY_COORDS.get(city_code, (None,None,city_code))[2]
    best      = ev_data["best_side"]
    side_data = ev_data[best]
    conf      = forecast["confidence"]
    bar       = {"high":"🟢🟢🟢","medium":"🟢🟢⚪","low":"🟢⚪⚪"}.get(conf,"⚪⚪⚪")
    t_ev      = side_data["taker_ev"]
    m_ev      = side_data["maker_ev"]
    color     = 0x00C851 if t_ev >= FIRE_EV_THRESHOLD*100 else 0xFFBB33
    fields = [
        {"name":"📍 City",           "value":city_name,                                         "inline":True},
        {"name":"📋 Market",         "value":market["subtitle"],                                 "inline":True},
        {"name":"🎯 Side",           "value":best,                                               "inline":True},
        {"name":"📊 Model prob",     "value":f"{side_data['prob']}%",                           "inline":True},
        {"name":"📋 Kalshi implied", "value":f"{side_data['implied']}%",                        "inline":True},
        {"name":"🔧 Bias correction","value":f"{forecast['bias_applied']:+.2f}°F",              "inline":True},
        {"name":"📈 EV% (taker)",    "value":f"+{t_ev}%",                                       "inline":True},
        {"name":"📈 EV% (maker)",    "value":f"+{m_ev}% ← post limit order",                   "inline":True},
        {"name":"💰 Half Kelly",     "value":f"{side_data['taker_hk']}% (taker) / {side_data['maker_hk']}% (maker)","inline":False},
        {"name":"💎 QK suggest",     "value":format_units(units),                               "inline":True},
        {"name":"🌡️ Ensemble mean",  "value":f"{forecast['corrected_mean']}°F (raw {forecast['ensemble_mean']}°F)","inline":True},
        {"name":"📐 Spread",         "value":f"±{forecast['spread']}°F",                        "inline":True},
        {"name":"🎯 Confidence",     "value":f"{bar} {conf.capitalize()}",                      "inline":True},
    ]
    if obs_high is not None:
        fields.append({"name":"🔴 ASOS observed high","value":f"{obs_high}°F today","inline":True})
    for key, label in [("ecmwf_high","🌍 ECMWF"),("hrrr_high","⚡ HRRR"),("nbm_high","🎯 NBM"),
                        ("rap_high","🔄 RAP"),("icon_high","🇩🇪 ICON"),("tomorrow_high","🤖 Tomorrow")]:
        if forecast.get(key):
            fields.append({"name":label,"value":f"{forecast[key]}°F","inline":True})
    total = forecast.get("total_members", 0)
    if total > 0:
        fields.append({"name":"📊 Ensemble members",
                        "value":f"{total} total ({forecast.get('gfs_members',0)} GFS + {forecast.get('ecmwf_members',0)} ECMWF)",
                        "inline":False})
    if tweet_hit: fields.append({"name":"🐦 Signal","value":"Met/NWS tweet","inline":True})
    if afd_hit:   fields.append({"name":"📄 AFD signal","value":"NWS forecast discussion flagged","inline":True})
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
    if series.startswith("KXLOWT"):   return None  # low temp needs separate min-temp forecast
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
_market_cache_ts: float = 0.0
MARKET_CACHE_TTL = 280

def get_active_kalshi_markets() -> list[dict]:
    global _market_cache, _market_cache_ts
    with _lock:
        cache_age = time.time() - _market_cache_ts
        if _market_cache and cache_age < MARKET_CACHE_TTL:
            raw = list(_market_cache)
            print(f"[kalshi] Using cached market list ({len(raw)} markets, {cache_age:.0f}s old)")
            markets = [m for m in (parse_market(r) for r in raw) if m]
            print(f"[kalshi] {len(markets)} active temperature markets (from cache)")
            return markets
    markets_raw = []
    for series in KXHIGH_SERIES:
        try:
            r = requests.get(f"{KALSHI_BASE}/markets",
                params={"status":"open","series_ticker":series,"limit":25}, timeout=10)
            if r.status_code == 429:
                print(f"[kalshi] Rate limited, waiting 30s...")
                time.sleep(30)
                r = requests.get(f"{KALSHI_BASE}/markets",
                    params={"status":"open","series_ticker":series,"limit":25}, timeout=10)
            r.raise_for_status()
            markets_raw.extend(r.json().get("markets", []))
        except Exception as e:
            print(f"[kalshi] {series}: {e}")
        time.sleep(0.5)
    markets = [m for m in (parse_market(r) for r in markets_raw) if m]
    print(f"[kalshi] {len(markets)} active temperature markets")
    return markets

# ── ASYNC SCAN ────────────────────────────────────────────────────────────────
async def scan_market_async(session, semaphore, market, today, tweet_cities, afd_cities):
    cc = market["city_code"]
    if cc not in CITY_COORDS: return None

    forecast = await get_forecast(session, semaphore, cc, today)
    if not forecast: return None

    threshold = market["threshold_f"]
    kind      = market["threshold_kind"]

    # v3.2: correct CDF-based probability for bucket vs threshold markets
    if kind == "B":
        lo   = threshold - 0.5
        hi   = threshold + 0.5
        prob = model_probability(forecast, threshold, cc, kind="B", lo=lo, hi=hi)
    else:
        lo, hi = None, None
        prob   = model_probability(forecast, threshold, cc, kind="T")

    implied_p = market["yes_price"] / 100
    adj       = longshot_probability_adjustment(implied_p)
    prob      = max(0.01, min(0.99, prob + adj))

    ev_data = compute_ev_kelly(prob, market["yes_price"], market["no_price"])
    best    = ev_data["best_side"]

    if not best: return None

    side_data = ev_data[best]
    t_ev      = side_data["taker_ev"]
    spread    = forecast["spread"]

    with _lock:
        obs_high = asos_observed.get(cc)
    tweet_hit = bool(tweet_cities and cc in tweet_cities)
    afd_hit   = bool(afd_cities and cc in afd_cities)

    # v3.3: log AFTER ev_data computed, with best_side + taker_ev
    if BIAS_LOGGING:
        try:
            _log_prediction(
                today, cc, CITY_COORDS[cc][2], market["ticker"], threshold,
                forecast, prob, market["yes_price"], market["no_price"],
                obs_high,
                best_side=best,
                taker_ev=ev_data[best]["taker_ev"],
                threshold_kind=kind,
            )
        except Exception as _e:
            print(f"[bias_logger] {_e}")

    fire  = t_ev >= FIRE_EV_THRESHOLD*100 and spread <= MAX_SPREAD_FIRE
    watch = t_ev >= WATCH_EV_THRESHOLD*100 and spread <= MAX_SPREAD_WATCH
    city  = CITY_COORDS[cc][2]
    tag   = "🔥" if fire else ("⚠️" if watch else ("🐦" if tweet_hit or afd_hit else "—"))
    print(f"[scan] {city}: model={side_data['prob']}% implied={side_data['implied']}% "
          f"EV={t_ev}% (maker {side_data['maker_ev']}%) spread=±{spread}°F {tag}")

    if fire or watch or ((tweet_hit or afd_hit) and t_ev > 0):
        return {"market":market,"forecast":forecast,"ev_data":ev_data,
                "obs_high":obs_high,"tweet":tweet_hit,"afd":afd_hit,"fire":fire}
    return None

async def run_scan_async(force_codes=None):
    ts = datetime.now(ET_TZ).strftime("%H:%M ET")
    markets = get_active_kalshi_markets()
    if not markets:
        msg = f"📊 **Scan done** {ts} | 0 temperature markets found"
        if force_codes: msg += f" | triggered: {', '.join(force_codes)}"
        post_discord(DISCORD_LOG_WEBHOOK, msg)
        print(f"[scan] Done — 0 markets found")
        return
    if force_codes:
        markets = [m for m in markets if m["city_code"] in force_codes]

    today = date.today()

    # Filter out stale markets (settlement date = today or earlier)
    def ticker_date(ticker):
        try:
            part = ticker.split("-")[1]  # e.g. 26MAY24
            months = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,
                      "JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
            yr  = 2000 + int(part[0:2])
            mon = months[part[2:5]]
            day = int(part[5:7])
            return date(yr, mon, day)
        except:
            return today
    markets = [m for m in markets if ticker_date(m["ticker"]) > today]
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT+4)
    timeout   = aiohttp.ClientTimeout(total=30)
    alerts    = 0

    with _lock:
        tweet_cities = set(tweet_flagged_cities)
        afd_cities   = set(afd_flagged_cities)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        results = await asyncio.gather(
            *[scan_market_async(session,semaphore,m,today,tweet_cities,afd_cities) for m in markets],
            return_exceptions=True,
        )

    for res in results:
        if isinstance(res, Exception) or res is None: continue
        market   = res["market"]
        forecast = res["forecast"]
        ev_data  = res["ev_data"]
        city     = CITY_COORDS.get(market["city_code"],(None,None,market["city_code"]))[2]
        best     = ev_data["best_side"]
        t_ev_res = ev_data[best]["taker_ev"]
        fire     = res["fire"]
        emoji    = "🔥" if fire else "⚠️"
        units    = recommend_units(t_ev_res, forecast["confidence"], res["tweet"], res["afd"], fire)
        embed    = build_embed(market,forecast,ev_data,res["obs_high"],res["tweet"],res["afd"],units)
        post_discord(DISCORD_WEBHOOK_URL, f"{emoji} **{city} — {market['subtitle']}**", [embed])
        alerts += 1

    msg = f"📊 **Scan done** {ts} | {len(markets)} markets | {alerts} alert(s)"
    if force_codes: msg += f" | triggered: {', '.join(force_codes)}"
    post_discord(DISCORD_LOG_WEBHOOK, msg)
    print(f"[scan] Done — {alerts} alert(s)")

def run_scan(force_codes=None):
    asyncio.run(run_scan_async(force_codes))

# ── TWITTER ───────────────────────────────────────────────────────────────────
def get_user_ids(usernames):
    if not X_BEARER_TOKEN: return {}
    hdr, ids = {"Authorization":f"Bearer {X_BEARER_TOKEN}"}, {}
    for i in range(0,len(usernames),100):
        try:
            r = requests.get("https://api.twitter.com/2/users/by",
                params={"usernames":",".join(usernames[i:i+100]),"user.fields":"id,username"},
                headers=hdr,timeout=10)
            r.raise_for_status()
            for u in r.json().get("data",[]):
                ids[u["username"].lower()] = u["id"]
        except Exception as e: print(f"[twitter] {e}")
    return ids

def classify_tweet(text) -> dict:
    if not ANTHROPIC_API_KEY:
        return {"is_signal":False,"cities":[],"direction":"","summary":""}
    sys_prompt = ("You classify meteorologist/NWS tweets for temperature market signals. "
           "Respond ONLY with valid JSON. SIGNAL = meaningful forecast CHANGE. "
           '{"is_signal":bool,"cities":["names"],"direction":"warmer"|"cooler"|"uncertain"|"",'
           '"confidence":"high"|"medium"|"low","summary":"one sentence or empty"}')
    try:
        r = requests.post("https://api.anthropic.com/v1/messages",
            headers={"x-api-key":ANTHROPIC_API_KEY,"anthropic-version":"2023-06-01","content-type":"application/json"},
            json={"model":"claude-sonnet-4-20250514","max_tokens":200,"system":sys_prompt,
                  "messages":[{"role":"user","content":f"Tweet: {text}"}]},timeout=15)
        r.raise_for_status()
        data = r.json()
        if "content" not in data or not data["content"]:
            return {"is_signal":False,"cities":[],"direction":"","summary":""}
        return json.loads(data["content"][0]["text"].strip())
    except json.JSONDecodeError:
        return {"is_signal":False,"cities":[],"direction":"","summary":""}
    except Exception as e:
        print(f"[claude] {e}")
        return {"is_signal":False,"cities":[],"direction":"","summary":""}

def fetch_timeline(uid, since_id):
    if not X_BEARER_TOKEN: return []
    hdr    = {"Authorization":f"Bearer {X_BEARER_TOKEN}"}
    params = {"max_results":10,"tweet.fields":"id,text,created_at","exclude":"retweets,replies"}
    if since_id: params["since_id"] = since_id
    try:
        r = requests.get(f"https://api.twitter.com/2/users/{uid}/tweets",
                         params=params,headers=hdr,timeout=10)
        r.raise_for_status()
        return r.json().get("data",[])
    except: return []

def tweet_scanner_loop(user_ids):
    print(f"[twitter] Monitoring {len(user_ids)} accounts")
    user_since = {uid: None for uid in user_ids.values()}
    while True:
        for username, uid in user_ids.items():
            for tweet in fetch_timeline(uid, user_since.get(uid)):
                tid = tweet["id"]
                if tid in seen_tweet_ids: continue
                seen_tweet_ids.add(tid)
                user_since[uid] = tid
                clf = classify_tweet(tweet["text"])
                time.sleep(1)
                if clf.get("is_signal"):
                    codes = names_to_codes(clf.get("cities",[]))
                    if codes:
                        with _lock: tweet_flagged_cities.update(codes)
                        print(f"[twitter] Signal @{username}: {codes} | {clf.get('direction')} | {clf.get('summary')}")
        time.sleep(TWEET_POLL_SECS)

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
                time.sleep(30)
                r = requests.get(f"{KALSHI_BASE}/markets",
                    params={"status":"open","series_ticker":series,"limit":25}, timeout=8)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[price_watcher] fetch error {series}: {e}")
            continue
        for m in data.get("markets", []):
            ticker    = m.get("ticker", "")
            city_code = series.replace("KXHIGH", "")
            yes_price = max(1, min(99, round(float(m.get("yes_ask_dollars") or 0.5) * 100)))
            no_price  = max(1, min(99, round(float(m.get("no_ask_dollars")  or 0.5) * 100)))
            if ticker:
                prices[ticker] = (yes_price, no_price, city_code)
                markets_raw.append(m)
        time.sleep(0.5)
    with _lock:
        _market_cache    = markets_raw
        _market_cache_ts = time.time()
    return prices

def price_watcher_loop():
    global price_snapshot
    print(f"[price_watcher] Starting — polling every {PRICE_POLL_SECS}s, trigger on {PRICE_MOVE_TRIGGER}¢ move")
    price_snapshot = {t: (y, n) for t, (y, n, _) in fetch_current_prices().items()}
    time.sleep(PRICE_POLL_SECS)
    while True:
        current = fetch_current_prices()
        moved_cities = set()
        for ticker, (yes_new, no_new, city_code) in current.items():
            prev = price_snapshot.get(ticker)
            if prev is None:
                price_snapshot[ticker] = (yes_new, no_new)
                continue
            yes_old, no_old = prev
            if abs(yes_new-yes_old) >= PRICE_MOVE_TRIGGER or abs(no_new-no_old) >= PRICE_MOVE_TRIGGER:
                moved_cities.add(city_code)
                direction = "↑ YES" if yes_new > yes_old else ("↓ YES" if yes_new < yes_old else "")
                print(f"[price_watcher] {city_code} {ticker}: yes {yes_old}¢→{yes_new}¢ no {no_old}¢→{no_new}¢ {direction}")
            price_snapshot[ticker] = (yes_new, no_new)
        if moved_cities:
            known = {c for c in moved_cities if c in CITY_COORDS}
            if known:
                print(f"[price_watcher] Price move detected → immediate rescan: {known}")
                run_scan(force_codes=known)
        time.sleep(PRICE_POLL_SECS)

def signal_rescan_loop():
    while True:
        time.sleep(60)
        with _lock:
            cities = set(tweet_flagged_cities) | set(afd_flagged_cities)
            if cities:
                tweet_flagged_cities.clear()
                afd_flagged_cities.clear()
        if cities:
            print(f"[rescan] Signal-triggered: {cities}")
            run_scan(force_codes=cities)

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
def main():
    print("🌡️  Kalshi Weather Bot v3.4")
    print(f"   v3.2: CDF bucket probabilities — buckets sum to 100%")
    print(f"   v3.3: bias_logger logs best_side + taker_ev correctly")
    print(f"   Concurrency:   {MAX_CONCURRENT} simultaneous model requests")
    print(f"   Price watcher: every {PRICE_POLL_SECS}s, trigger on {PRICE_MOVE_TRIGGER}¢ move")
    print(f"   Accounts:      {len(ALL_ACCOUNTS)} X accounts monitored")
    print(f"   WFO offices:   {len(set(info[4] for info in CITY_COORDS.values()))} AFDs polled")

    if not DISCORD_WEBHOOK_URL: print("[warn] DISCORD_WEBHOOK_URL not set")
    if not X_BEARER_TOKEN:      print("[warn] X_BEARER_TOKEN not set")
    if not ANTHROPIC_API_KEY:   print("[warn] ANTHROPIC_API_KEY not set")

    threading.Thread(target=asos_poll_loop,    daemon=True).start()
    threading.Thread(target=afd_scanner_loop,  daemon=True).start()

    user_ids = {}
    if X_BEARER_TOKEN:
        print(f"[twitter] Resolving {len(ALL_ACCOUNTS)} account IDs...")
        user_ids = get_user_ids(ALL_ACCOUNTS)
        print(f"[twitter] Resolved {len(user_ids)}/{len(ALL_ACCOUNTS)}")
    if user_ids:
        threading.Thread(target=tweet_scanner_loop, args=(user_ids,), daemon=True).start()

    threading.Thread(target=price_watcher_loop, daemon=True).start()
    threading.Thread(target=signal_rescan_loop, daemon=True).start()

    print("[main] Waiting 100s for price watcher to populate market cache...")
    time.sleep(100)

    while True:
        try:
            now_et = datetime.now(ET_TZ)
            if now_et.weekday() == 3 and 3 <= now_et.hour < 5:
                print(f"[main] Kalshi maintenance window (Thu 3-5am ET) — sleeping 30 min")
                time.sleep(1800)
                continue
            run_scan()
        except Exception as e:
            print(f"[error] {e}")
        time.sleep(SCAN_INTERVAL_SECS)

if __name__ == "__main__":
    main()
