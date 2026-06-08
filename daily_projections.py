"""
daily_projections.py — Next-day temperature projections for all cities

Pulls all weather models and posts tomorrow's predicted high for each city
to Discord. Run manually whenever you want to see projections.

Usage: python3 daily_projections.py
Environment vars needed: DISCORD_PROJECTION_WEBHOOK
"""

import os, asyncio, aiohttp, math, time, requests
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

DISCORD_PROJECTION_WEBHOOK = os.environ.get("DISCORD_PROJECTION_WEBHOOK", "")
TOMORROW_IO_KEY            = os.environ.get("TOMORROW_IO_KEY", "")

OPEN_METEO_BASE     = "https://api.open-meteo.com/v1"
OPEN_METEO_ENS_BASE = "https://ensemble-api.open-meteo.com/v1"

ET_TZ = ZoneInfo("America/New_York")

# ── CITY CONFIG ───────────────────────────────────────────────────────────────
CITIES = {
    "NY":  (40.7128,  -74.0060, "New York City",    "ET"),
    "AUS": (30.2672,  -97.7431, "Austin",           "CT"),
    "LAX": (34.0522, -118.2437, "Los Angeles",      "PT"),
    "CHI": (41.8781,  -87.6298, "Chicago",          "CT"),
    "MIA": (25.7617,  -80.1918, "Miami",            "ET"),
    "DAL": (32.7767,  -96.7970, "Dallas",           "CT"),
    "DC":  (38.9072,  -77.0369, "Washington DC",    "ET"),
    "SEA": (47.6062, -122.3321, "Seattle",          "PT"),
    "PHX": (33.4484, -112.0740, "Phoenix",          "MT"),
    "BOS": (42.3601,  -71.0589, "Boston",           "ET"),
    "HOU": (29.7604,  -95.3698, "Houston",          "CT"),
    "ATL": (33.7490,  -84.3880, "Atlanta",          "ET"),
    "OKC": (35.4676,  -97.5164, "Oklahoma City",    "CT"),
    "LV":  (36.1699, -115.1398, "Las Vegas",        "PT"),
    "SFO": (37.7749, -122.4194, "San Francisco",    "PT"),
    "DEN": (39.7392, -104.9903, "Denver",           "MT"),
    "SA":  (29.4241,  -98.4936, "San Antonio",      "CT"),
    "MN":  (44.9778,  -93.2650, "Minneapolis",      "CT"),
    "SD":  (32.7157, -117.1611, "San Diego",        "PT"),
    "KC":  (39.0997,  -94.5786, "Kansas City",      "CT"),
    "SL":  (38.6270,  -90.1994, "St. Louis",        "CT"),
    "PO":  (45.5051, -122.6750, "Portland",         "PT"),
    "NO":  (29.9511,  -90.0715, "New Orleans",      "CT"),
    "MEM": (35.1495,  -90.0490, "Memphis",          "CT"),
}

# ── BIAS CORRECTIONS ──────────────────────────────────────────────────────────
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

# ── FORECAST FETCHERS ─────────────────────────────────────────────────────────
async def _get_json(session, url, params):
    async with session.get(url, params=params) as r:
        return await r.json()

async def fetch_det_models(session, lat, lon, ds):
    results = {}
    try:
        data = await _get_json(session, f"{OPEN_METEO_BASE}/ecmwf", {
            "latitude": lat, "longitude": lon, "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit", "start_date": ds, "end_date": ds})
        temps = [t for t in (data.get("hourly", {}).get("temperature_2m") or []) if t is not None]
        if temps: results["ECMWF"] = max(temps)
    except: pass

    try:
        data = await _get_json(session, f"{OPEN_METEO_BASE}/gfs", {
            "latitude": lat, "longitude": lon, "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit", "start_date": ds, "end_date": ds,
            "models": "ncep_hrrr_conus"})
        temps = [t for t in (data.get("hourly", {}).get("temperature_2m") or []) if t is not None]
        if temps: results["HRRR"] = max(temps)
    except: pass

    try:
        data = await _get_json(session, f"{OPEN_METEO_BASE}/forecast", {
            "latitude": lat, "longitude": lon, "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit", "start_date": ds, "end_date": ds,
            "models": "ncep_nbm_conus"})
        temps = [t for t in (data.get("hourly", {}).get("temperature_2m") or []) if t is not None]
        if temps: results["NBM"] = max(temps)
    except: pass

    try:
        data = await _get_json(session, f"{OPEN_METEO_BASE}/forecast", {
            "latitude": lat, "longitude": lon, "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit", "start_date": ds, "end_date": ds,
            "models": "icon_seamless"})
        temps = [t for t in (data.get("hourly", {}).get("temperature_2m") or []) if t is not None]
        if temps: results["ICON"] = max(temps)
    except: pass

    if TOMORROW_IO_KEY:
        try:
            async with session.get("https://api.tomorrow.io/v4/weather/forecast",
                params={"location": f"{lat},{lon}", "apikey": TOMORROW_IO_KEY,
                        "units": "imperial", "timesteps": "1d", "fields": "temperatureMax",
                        "startTime": f"{ds}T00:00:00Z", "endTime": f"{ds}T23:59:59Z"}) as r:
                data = await r.json()
                days = data.get("timelines", {}).get("daily", [])
                if days: results["Tomorrow"] = days[0].get("values", {}).get("temperatureMax")
        except: pass

    return results

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

async def get_city_projection(session, semaphore, city_code, target_date):
    lat, lon, name, tz = CITIES[city_code]
    ds = target_date.isoformat()

    async with semaphore:
        det_task = fetch_det_models(session, lat, lon, ds)
        ens_task  = fetch_ensemble_members(session, lat, lon, ds)
        det, ens  = await asyncio.gather(det_task, ens_task, return_exceptions=True)

    if isinstance(det, Exception): det = {}
    if isinstance(ens, Exception): ens = []

    if not det and not ens:
        return None

    # Build blend
    blend = list(ens)
    if det.get("NBM"):    blend += [det["NBM"]] * 5
    if det.get("ECMWF"):  blend += [det["ECMWF"], det["ECMWF"]]
    if det.get("HRRR"):   blend += [det["HRRR"], det["HRRR"]]
    if det.get("ICON"):   blend.append(det["ICON"])
    if det.get("Tomorrow"): blend.append(det["Tomorrow"])

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

    # Widen spread if models disagree
    det_vals = [v for v in det.values() if v is not None]
    if len(det_vals) >= 2:
        det_range = max(det_vals) - min(det_vals)
        if det_range > 4.0:
            spread = max(spread, det_range / 2)

    bias = get_bias(city_code)
    corrected_mean = mean + bias

    conf = "🟢" if spread < 2.0 else ("🟡" if spread < 4.0 else "🔴")

    model_strs = []
    for key, label in [("ECMWF","ECMWF"),("HRRR","HRRR"),("NBM","NBM"),
                        ("ICON","ICON"),("Tomorrow","TMR")]:
        if det.get(key):
            model_strs.append(f"{label} {det[key]:.0f}°")

    return {
        "city_code":      city_code,
        "city_name":      name,
        "mean":           round(corrected_mean, 1),
        "spread":         round(spread, 2),
        "conf":           conf,
        "models":         model_strs,
        "ens_members":    len(ens),
        "det_count":      len(det_vals),
    }

# ── MAIN ──────────────────────────────────────────────────────────────────────
async def run():
    tomorrow = date.today() + timedelta(days=1)
    now_et   = datetime.now(ET_TZ)
    ts       = now_et.strftime("%I:%M %p ET")

    print(f"Running projections for {tomorrow} at {ts}")

    semaphore = asyncio.Semaphore(8)
    connector = aiohttp.TCPConnector(limit=12)
    timeout   = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [get_city_projection(session, semaphore, code, tomorrow)
                 for code in CITIES]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    valid = [r for r in results if r and not isinstance(r, Exception)]
    valid.sort(key=lambda x: -x["mean"])  # hottest first

    if not valid:
        print("No projections available")
        return

    # Build embed lines
    lines = []
    for r in valid:
        model_str = " | ".join(r["models"][:4]) if r["models"] else "no models"
        lines.append(
            f"{r['conf']} **{r['city_name']}** — `{r['mean']}°F` ±{r['spread']}°F\n"
            f"  {model_str} | {r['ens_members']} ens members"
        )

    # Split into chunks of 8 cities per embed
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
            "footer": {"text": f"Generated {ts} | 🟢 High conf ±<2°F | 🟡 Med ±2-4°F | 🔴 Low ±4°F+"}
        })

    if DISCORD_PROJECTION_WEBHOOK:
        for embed in embeds:
            requests.post(DISCORD_PROJECTION_WEBHOOK,
                          json={"content": "", "embeds": [embed]},
                          timeout=10)
            time.sleep(1)
        print(f"Posted {len(valid)} city projections to Discord")
    else:
        print("[warn] DISCORD_PROJECTION_WEBHOOK not set")
        for r in valid:
            print(f"{r['city_name']}: {r['mean']}°F ±{r['spread']}°F")

if __name__ == "__main__":
    asyncio.run(run())
