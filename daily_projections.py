"""
daily_projections.py — Next-day temperature projections for all cities

Pulls the same Wethr.net model stack the main bot uses (HRRR, NBM, RAP,
NAM4KM, NWS, ECMWF-IFS) plus Open-Meteo ensembles for spread, applies the
same bias corrections, and posts tomorrow's predicted high for each city.

Runs 3x daily via cron, or manually.

Usage: python3 daily_projections.py
Environment vars needed: DISCORD_PROJECTION_WEBHOOK, WETHR_API_KEY
"""

import os, asyncio, aiohttp, math, time, requests
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

DISCORD_PROJECTION_WEBHOOK = os.environ.get("DISCORD_PROJECTION_WEBHOOK", "")
WETHR_API_KEY              = os.environ.get("WETHR_API_KEY", "")
TOMORROW_IO_KEY            = os.environ.get("TOMORROW_IO_KEY", "")

OPEN_METEO_ENS_BASE = "https://ensemble-api.open-meteo.com/v1"
WETHR_FORECAST_BASE = "https://wethr.net/api/v2/forecasts.php"
WETHR_NWS_BASE      = "https://wethr.net/api/v2/nws_forecasts.php"
WETHR_MODELS        = ["HRRR", "NBM", "RAP", "NAM4KM", "GFS", "ECMWF-IFS"]

ET_TZ = ZoneInfo("America/New_York")

# ── CITY CONFIG (coords + Wethr station) ─────────────────────────────────────
CITIES = {
    "NY":  (40.7128,  -74.0060, "New York City",   "KNYC"),
    "AUS": (30.2672,  -97.7431, "Austin",          "KAUS"),
    "LAX": (34.0522, -118.2437, "Los Angeles",     "KLAX"),
    "CHI": (41.8781,  -87.6298, "Chicago",         "KMDW"),
    "MIA": (25.7617,  -80.1918, "Miami",           "KMIA"),
    "DAL": (32.7767,  -96.7970, "Dallas",          "KDFW"),
    "DC":  (38.9072,  -77.0369, "Washington DC",   "KDCA"),
    "SEA": (47.6062, -122.3321, "Seattle",         "KSEA"),
    "PHX": (33.4484, -112.0740, "Phoenix",         "KPHX"),
    "BOS": (42.3601,  -71.0589, "Boston",          "KBOS"),
    "HOU": (29.7604,  -95.3698, "Houston",         "KHOU"),
    "ATL": (33.7490,  -84.3880, "Atlanta",         "KATL"),
    "OKC": (35.4676,  -97.5164, "Oklahoma City",   "KOKC"),
    "LV":  (36.1699, -115.1398, "Las Vegas",       "KLAS"),
    "SFO": (37.7749, -122.4194, "San Francisco",   "KSFO"),
    "DEN": (39.7392, -104.9903, "Denver",          "KDEN"),
    "SA":  (29.4241,  -98.4936, "San Antonio",     "KSAT"),
    "MN":  (44.9778,  -93.2650, "Minneapolis",     "KMSP"),
    "NO":  (29.9511,  -90.0715, "New Orleans",     "KMSY"),
}

# ── BIAS CORRECTIONS (same as main bot v3.27) ────────────────────────────────
CITY_BIAS_F = {
    "NY":  [ 0.8,  0.7,  0.5,  0.3,  0.2,  0.0, -0.3, -0.2,  0.0,  0.3,  0.5,  0.7],
    "CHI": [ 1.2,  1.0,  0.8,  0.4,  0.2,  0.0, -0.5, -0.4,  0.0,  0.5,  0.8,  1.1],
    "LAX": [-0.5, -0.4, -0.3, -0.2, -0.2, -0.3, -0.4, -0.4, -0.3, -0.2, -0.3, -0.4],
    "MIA": [ 0.3,  0.3,  0.2,  0.1,  0.0,  2.5,  2.5,  2.5,  0.0,  0.1,  0.2,  0.3],
    "PHX": [-0.4, -0.3, -0.2, -0.1,  0.0, -0.3, -0.8, -0.7, -0.3, -0.1, -0.2, -0.3],
    "DEN": [ 0.6,  0.5,  0.4,  0.2,  0.1,  0.0, -0.4, -0.3,  0.0,  0.3,  0.5,  0.6],
    "LV":  [-0.5, -0.4, -0.2,  0.0,  0.1, -0.2, -0.8, -0.7, -0.2,  0.0, -0.2, -0.4],
    "SEA": [-0.3, -0.3, -0.2, -0.1,  0.0,  0.0, -0.2, -0.2, -0.1,  0.0, -0.2, -0.3],
    "HOU": [ 0.2,  0.1,  0.0, -0.1, -0.2, -0.4, -0.6, -0.6, -0.3, -0.1,  0.1,  0.2],
}
DEFAULT_BIAS = [0.0] * 12

def get_bias(city_code):
    month = datetime.now().month - 1
    return CITY_BIAS_F.get(city_code, DEFAULT_BIAS)[month]

# ── WETHR FORECAST FETCHERS (blocking, with 429 retry) ───────────────────────
def fetch_wethr_forecast_high(station, model, target_date):
    if not WETHR_API_KEY:
        return None
    try:
        r = requests.get(WETHR_FORECAST_BASE, params={
            "location_name": station, "model": model, "run": "latest",
        }, headers={"Authorization": f"Bearer {WETHR_API_KEY}"}, timeout=10)
        if r.status_code == 429:
            time.sleep(10)
            r = requests.get(WETHR_FORECAST_BASE, params={
                "location_name": station, "model": model, "run": "latest",
            }, headers={"Authorization": f"Bearer {WETHR_API_KEY}"}, timeout=10)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            return None
        highs = []
        for row in rows:
            if row.get("valid_time", "")[:10] == target_date:
                t = row.get("temperature_f")
                if t is not None:
                    highs.append(float(t))
        return max(highs) if highs else None
    except Exception as e:
        print(f"[wethr_fcst] {station}/{model}: {e}")
        return None

def fetch_wethr_nws_forecast(station, target_date):
    if not WETHR_API_KEY:
        return None
    try:
        r = requests.get(WETHR_NWS_BASE, params={
            "station_code": station, "date": target_date, "mode": "latest",
        }, headers={"Authorization": f"Bearer {WETHR_API_KEY}"}, timeout=10)
        if r.status_code == 429:
            time.sleep(10)
            r = requests.get(WETHR_NWS_BASE, params={
                "station_code": station, "date": target_date, "mode": "latest",
            }, headers={"Authorization": f"Bearer {WETHR_API_KEY}"}, timeout=10)
        r.raise_for_status()
        data = r.json()
        high = data.get("high")
        return float(high) if high is not None else None
    except Exception as e:
        print(f"[wethr_nws] {station}: {e}")
        return None

def fetch_wethr_all_models(station, target_date):
    if not WETHR_API_KEY or not station:
        return {}
    results = {}
    for model in WETHR_MODELS:
        high = fetch_wethr_forecast_high(station, model, target_date)
        if high is not None:
            results[model] = high
        time.sleep(1.5)
    nws = fetch_wethr_nws_forecast(station, target_date)
    if nws is not None:
        results["NWS"] = nws
    return results

# ── OPEN-METEO ENSEMBLE (for spread) ─────────────────────────────────────────
async def _get_json(session, url, params):
    async with session.get(url, params=params) as r:
        return await r.json()

async def fetch_ensemble_members(session, lat, lon, ds):
    members = []
    try:
        keys = ",".join([f"temperature_2m_member{i:02d}" for i in range(1, 32)])
        data = await _get_json(session, f"{OPEN_METEO_ENS_BASE}/ensemble", {
            "latitude": lat, "longitude": lon, "hourly": keys,
            "temperature_unit": "fahrenheit", "start_date": ds, "end_date": ds,
            "models": "gfs_seamless"})
        hourly = data.get("hourly", {})
        for i in range(1, 32):
            temps = [t for t in (hourly.get(f"temperature_2m_member{i:02d}") or []) if t is not None]
            if temps: members.append(max(temps))
    except: pass

    try:
        keys = ",".join([f"temperature_2m_member{i:02d}" for i in range(0, 51)])
        data = await _get_json(session, f"{OPEN_METEO_ENS_BASE}/ensemble", {
            "latitude": lat, "longitude": lon, "hourly": keys,
            "temperature_unit": "fahrenheit", "start_date": ds, "end_date": ds,
            "models": "ecmwf_ifs025"})
        hourly = data.get("hourly", {})
        for i in range(0, 51):
            temps = [t for t in (hourly.get(f"temperature_2m_member{i:02d}") or []) if t is not None]
            if temps: members.append(max(temps))
    except: pass

    return members

def build_projection(city_code, wethr_models, ens, target_date):
    """Combine Wethr models + ensemble into a projection (same logic as bot)."""
    lat, lon, name, station = CITIES[city_code]

    det = dict(wethr_models)
    if not det and not ens:
        return None

    # Blend — same weights as main bot
    blend = list(ens)
    if det.get("NBM"):       blend += [det["NBM"]] * 5
    if det.get("NWS"):       blend += [det["NWS"]] * 3
    if det.get("ECMWF-IFS"): blend += [det["ECMWF-IFS"], det["ECMWF-IFS"]]
    if det.get("HRRR"):      blend += [det["HRRR"], det["HRRR"]]
    if det.get("RAP"):       blend += [det["RAP"], det["RAP"]]
    if det.get("NAM4KM"):    blend.append(det["NAM4KM"])
    if det.get("GFS"):       blend.append(det["GFS"])

    if not blend:
        return None

    mean = sum(blend) / len(blend)

    if len(ens) >= 2:
        am = sum(ens) / len(ens)
        spread = math.sqrt(sum((x - am) ** 2 for x in ens) / len(ens))
    elif len(blend) >= 2:
        spread = math.sqrt(sum((x - mean) ** 2 for x in blend) / len(blend))
    else:
        spread = 3.0

    det_vals = [v for v in det.values() if v is not None]
    if len(det_vals) >= 2:
        det_range = max(det_vals) - min(det_vals)
        if det_range > 4.0:
            spread = max(spread, det_range / 2)

    bias = get_bias(city_code)
    corrected_mean = mean + bias

    conf = "🟢" if spread < 2.0 else ("🟡" if spread < 4.0 else "🔴")

    model_strs = []
    for key, label in [("HRRR","HRRR"),("NBM","NBM"),("RAP","RAP"),
                       ("NAM4KM","NAM"),("NWS","NWS"),("ECMWF-IFS","ECMWF")]:
        if det.get(key):
            model_strs.append(f"{label} {det[key]:.0f}°")

    return {
        "city_code":   city_code,
        "city_name":   name,
        "mean":        round(corrected_mean, 1),
        "spread":      round(spread, 2),
        "conf":        conf,
        "models":      model_strs,
        "ens_members": len(ens),
        "det_count":   len(det_vals),
    }

# ── MAIN ──────────────────────────────────────────────────────────────────────
async def run():
    tomorrow = date.today() + timedelta(days=1)
    ds       = tomorrow.isoformat()
    now_et   = datetime.now(ET_TZ)
    ts       = now_et.strftime("%I:%M %p ET")

    print(f"Running projections for {tomorrow} at {ts}")

    # 1) Fetch Wethr models sequentially (blocking, rate-limited)
    wethr_by_city = {}
    if WETHR_API_KEY:
        for code in CITIES:
            station = CITIES[code][3]
            wethr_by_city[code] = fetch_wethr_all_models(station, ds)
            print(f"  {code}: wethr={len(wethr_by_city[code])} models")

    # 2) Fetch Open-Meteo ensembles concurrently
    connector = aiohttp.TCPConnector(limit=12)
    timeout   = aiohttp.ClientTimeout(total=30)
    ens_by_city = {}
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        sem = asyncio.Semaphore(8)
        async def fetch_one(code):
            lat, lon, _, _ = CITIES[code]
            async with sem:
                ens = await fetch_ensemble_members(session, lat, lon, ds)
            return code, ens
        results = await asyncio.gather(*[fetch_one(c) for c in CITIES],
                                       return_exceptions=True)
        for r in results:
            if not isinstance(r, Exception):
                code, ens = r
                ens_by_city[code] = ens

    # 3) Build projections
    valid = []
    for code in CITIES:
        proj = build_projection(code, wethr_by_city.get(code, {}),
                                ens_by_city.get(code, []), ds)
        if proj:
            valid.append(proj)

    valid.sort(key=lambda x: -x["mean"])  # hottest first

    if not valid:
        print("No projections available")
        return

    lines = []
    for r in valid:
        model_str = " | ".join(r["models"][:4]) if r["models"] else "no models"
        lines.append(
            f"{r['conf']} **{r['city_name']}** — `{r['mean']}°F` ±{r['spread']}°F\n"
            f"  {model_str} | {r['ens_members']} ens members"
        )

    chunk_size = 12
    embeds = []
    for i in range(0, len(lines), chunk_size):
        chunk = lines[i:i + chunk_size]
        part  = i // chunk_size + 1
        total = math.ceil(len(lines) / chunk_size)
        embeds.append({
            "title": f"🌡️ Tomorrow's Projections — {tomorrow} (part {part}/{total})",
            "color": 0x5865F2,
            "description": "\n".join(chunk),
            "footer": {"text": f"Generated {ts} | 🟢 ±<2°F | 🟡 ±2-4°F | 🔴 ±4°F+ | Wethr models"}
        })

    if DISCORD_PROJECTION_WEBHOOK:
        for embed in embeds:
            requests.post(DISCORD_PROJECTION_WEBHOOK,
                          json={"content": "", "embeds": [embed]}, timeout=10)
            time.sleep(1)
        print(f"Posted {len(valid)} city projections to Discord")
    else:
        print("[warn] DISCORD_PROJECTION_WEBHOOK not set")
        for r in valid:
            print(f"{r['city_name']}: {r['mean']}°F ±{r['spread']}°F")

if __name__ == "__main__":
    asyncio.run(run())
