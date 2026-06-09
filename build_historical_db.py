"""
build_historical_db.py — Build historical temperature database from Open-Meteo

Pulls 5 years of hourly temperature data for all Kalshi cities and stores:
  1. Daily actual highs per city (for bias correction)
  2. Hourly temperature curves (for heating profile calculation)
  3. Model vs actual comparisons (for future bias tracking)

All data stored in PostgreSQL so bot can query in real-time.

Run once: python3 build_historical_db.py
Takes ~15-20 minutes to process all cities.
"""

import os, requests, psycopg2, json, time
from datetime import date, timedelta
from collections import defaultdict

DATABASE_URL     = os.environ.get("DATABASE_URL", "")
OPEN_METEO_BASE  = "https://archive-api.open-meteo.com/v1/archive"

# ── CITY CONFIG (station → coords) ───────────────────────────────────────────
CITIES = {
    "KNYC": (40.7128,  -74.0060, "New York City"),
    "KAUS": (30.2672,  -97.7431, "Austin"),
    "KLAX": (34.0522, -118.2437, "Los Angeles"),
    "KMDW": (41.8781,  -87.6298, "Chicago"),
    "KMIA": (25.7617,  -80.1918, "Miami"),
    "KDFW": (32.7767,  -96.7970, "Dallas"),
    "KDCA": (38.9072,  -77.0369, "Washington DC"),
    "KSEA": (47.6062, -122.3321, "Seattle"),
    "KPHX": (33.4484, -112.0740, "Phoenix"),
    "KBOS": (42.3601,  -71.0589, "Boston"),
    "KHOU": (29.7604,  -95.3698, "Houston"),
    "KATL": (33.7490,  -84.3880, "Atlanta"),
    "KOKC": (35.4676,  -97.5164, "Oklahoma City"),
    "KLAS": (36.1699, -115.1398, "Las Vegas"),
    "KSFO": (37.7749, -122.4194, "San Francisco"),
    "KDEN": (39.7392, -104.9903, "Denver"),
    "KSAT": (29.4241,  -98.4936, "San Antonio"),
    "KMSY": (29.9511,  -90.0715, "New Orleans"),
    "KMSP": (44.9778,  -93.2650, "Minneapolis"),
    "KSAN": (32.7157, -117.1611, "San Diego"),
}

# UTC offsets for local time conversion (standard time)
STATION_UTC_OFFSET = {
    "KNYC": -5, "KAUS": -6, "KLAX": -8, "KMDW": -6, "KMIA": -5,
    "KDFW": -6, "KDCA": -5, "KSEA": -8, "KPHX": -7, "KBOS": -5,
    "KHOU": -6, "KATL": -5, "KOKC": -6, "KLAS": -8, "KSFO": -8,
    "KDEN": -7, "KSAT": -6, "KMSY": -6, "KMSP": -6, "KSAN": -8,
}

def get_conn():
    return psycopg2.connect(DATABASE_URL)

def ensure_schema():
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS historical_highs (
                id              SERIAL PRIMARY KEY,
                station_code    TEXT NOT NULL,
                city_name       TEXT NOT NULL,
                date            DATE NOT NULL,
                month           INTEGER NOT NULL,
                actual_high_f   REAL NOT NULL,
                UNIQUE(station_code, date)
            );
            CREATE INDEX IF NOT EXISTS idx_hist_station ON historical_highs(station_code);
            CREATE INDEX IF NOT EXISTS idx_hist_date    ON historical_highs(date);
            CREATE INDEX IF NOT EXISTS idx_hist_month   ON historical_highs(month);

            CREATE TABLE IF NOT EXISTS heating_profiles_db (
                id              SERIAL PRIMARY KEY,
                station_code    TEXT NOT NULL,
                city_name       TEXT NOT NULL,
                month           INTEGER NOT NULL,
                avg_peak_hour   REAL,
                avg_rise_2pm    REAL,
                avg_rise_3pm    REAL,
                avg_rise_4pm    REAL,
                avg_rise_5pm    REAL,
                pct_rising_2pm  REAL,
                pct_rising_3pm  REAL,
                pct_rising_4pm  REAL,
                pct_rising_5pm  REAL,
                days_analyzed   INTEGER,
                updated_at      TIMESTAMP DEFAULT NOW(),
                UNIQUE(station_code, month)
            );
            CREATE INDEX IF NOT EXISTS idx_heat_station ON heating_profiles_db(station_code);
        """)
        conn.commit()
        print("[schema] Tables created/verified")
    finally:
        conn.close()

def fetch_historical_hourly(lat, lon, start_year=2021):
    """Fetch hourly temperature data from Open-Meteo archive API."""
    end_date   = date.today() - timedelta(days=5)  # archive has ~5 day lag
    start_date = date(start_year, 1, 1)

    try:
        r = requests.get(OPEN_METEO_BASE, params={
            "latitude":          lat,
            "longitude":         lon,
            "hourly":            "temperature_2m",
            "temperature_unit":  "fahrenheit",
            "start_date":        start_date.isoformat(),
            "end_date":          end_date.isoformat(),
            "timezone":          "UTC",
        }, timeout=60)
        r.raise_for_status()
        data = r.json()
        times = data.get("hourly", {}).get("time", [])
        temps = data.get("hourly", {}).get("temperature_2m", [])
        print(f"  Fetched {len(times)} hourly records ({start_date} to {end_date})")
        return list(zip(times, temps))
    except Exception as e:
        print(f"  [error] fetch failed: {e}")
        return []

def process_daily_highs(records, station_code, utc_offset):
    """
    Convert hourly UTC records to local-time daily highs.
    Uses NWS standard time convention (no DST) matching Kalshi settlement.
    """
    # Group by local standard date
    day_temps = defaultdict(list)

    for time_str, temp in records:
        if temp is None:
            continue
        try:
            # Parse UTC hour
            dt_str  = time_str[:16]  # "2021-01-01T00:00"
            yr      = int(dt_str[0:4])
            mo      = int(dt_str[5:7])
            dy      = int(dt_str[8:10])
            hr_utc  = int(dt_str[11:13])

            # Convert to local standard time
            hr_local = hr_utc + utc_offset
            local_date = date(yr, mo, dy)
            if hr_local < 0:
                local_date = local_date - timedelta(days=1)
                hr_local   += 24
            elif hr_local >= 24:
                local_date = local_date + timedelta(days=1)
                hr_local   -= 24

            day_temps[(local_date, hr_local)].append(float(temp))
        except Exception:
            continue

    # Get daily high (max temp per local date, hours 0-23)
    daily_highs = {}
    date_hours  = defaultdict(dict)
    for (local_date, hr), temps in day_temps.items():
        date_hours[local_date][hr] = max(temps)

    for local_date, hours in date_hours.items():
        if len(hours) >= 12:  # only use days with sufficient data
            daily_highs[local_date] = max(hours.values())

    return daily_highs, date_hours

def calculate_heating_profiles(date_hours, station_code, city_name):
    """Calculate heating profiles from hourly temperature curves."""
    profiles = {}

    for month in range(1, 13):
        month_days = [d for d in date_hours if d.month == month]
        if len(month_days) < 10:
            continue

        peak_hours  = []
        rise_2pm    = []
        rise_3pm    = []
        rise_4pm    = []
        rise_5pm    = []

        for day in month_days:
            curve = date_hours[day]
            if len(curve) < 12:
                continue

            # Find peak hour (10am-8pm local)
            peak_temp = peak_hour = None
            for hr in range(10, 21):
                t = curve.get(hr)
                if t is not None and (peak_temp is None or t > peak_temp):
                    peak_temp, peak_hour = t, hr

            if peak_temp is None:
                continue

            peak_hours.append(peak_hour)

            for hr, lst in [(14, rise_2pm), (15, rise_3pm),
                            (16, rise_4pm), (17, rise_5pm)]:
                t = curve.get(hr)
                if t is not None and peak_temp > t:
                    lst.append(round(peak_temp - t, 1))

        def avg(lst): return round(sum(lst)/len(lst), 2) if lst else 0.0
        def pct(lst): return round(len(lst)/max(len(month_days),1)*100, 1)

        profiles[month] = {
            "avg_peak_hour":  round(avg(peak_hours), 1),
            "avg_rise_2pm":   avg(rise_2pm),
            "avg_rise_3pm":   avg(rise_3pm),
            "avg_rise_4pm":   avg(rise_4pm),
            "avg_rise_5pm":   avg(rise_5pm),
            "pct_rising_2pm": pct(rise_2pm),
            "pct_rising_3pm": pct(rise_3pm),
            "pct_rising_4pm": pct(rise_4pm),
            "pct_rising_5pm": pct(rise_5pm),
            "days_analyzed":  len(month_days),
        }

    return profiles

def save_daily_highs(station_code, city_name, daily_highs):
    """Save daily highs to PostgreSQL."""
    if not daily_highs:
        return 0

    conn = get_conn()
    saved = 0
    try:
        cur = conn.cursor()
        for dt, high in daily_highs.items():
            cur.execute("""
                INSERT INTO historical_highs
                    (station_code, city_name, date, month, actual_high_f)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (station_code, date) DO UPDATE
                SET actual_high_f = EXCLUDED.actual_high_f
            """, (station_code, city_name, dt, dt.month, round(high, 1)))
            saved += 1
        conn.commit()
    finally:
        conn.close()
    return saved

def save_heating_profiles(station_code, city_name, profiles):
    """Save heating profiles to PostgreSQL."""
    if not profiles:
        return

    conn = get_conn()
    try:
        cur = conn.cursor()
        for month, p in profiles.items():
            cur.execute("""
                INSERT INTO heating_profiles_db
                    (station_code, city_name, month,
                     avg_peak_hour, avg_rise_2pm, avg_rise_3pm,
                     avg_rise_4pm, avg_rise_5pm,
                     pct_rising_2pm, pct_rising_3pm,
                     pct_rising_4pm, pct_rising_5pm,
                     days_analyzed, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
                ON CONFLICT (station_code, month) DO UPDATE SET
                    avg_peak_hour  = EXCLUDED.avg_peak_hour,
                    avg_rise_2pm   = EXCLUDED.avg_rise_2pm,
                    avg_rise_3pm   = EXCLUDED.avg_rise_3pm,
                    avg_rise_4pm   = EXCLUDED.avg_rise_4pm,
                    avg_rise_5pm   = EXCLUDED.avg_rise_5pm,
                    pct_rising_2pm = EXCLUDED.pct_rising_2pm,
                    pct_rising_3pm = EXCLUDED.pct_rising_3pm,
                    pct_rising_4pm = EXCLUDED.pct_rising_4pm,
                    pct_rising_5pm = EXCLUDED.pct_rising_5pm,
                    days_analyzed  = EXCLUDED.days_analyzed,
                    updated_at     = NOW()
            """, (
                station_code, city_name, month,
                p["avg_peak_hour"],
                p["avg_rise_2pm"], p["avg_rise_3pm"],
                p["avg_rise_4pm"], p["avg_rise_5pm"],
                p["pct_rising_2pm"], p["pct_rising_3pm"],
                p["pct_rising_4pm"], p["pct_rising_5pm"],
                p["days_analyzed"],
            ))
        conn.commit()
        print(f"  Saved {len(profiles)} monthly heating profiles to DB")
    finally:
        conn.close()

def print_june_summary(station_code, city_name, profiles):
    """Print June heating profile for verification."""
    p = profiles.get(6)
    if not p:
        return
    print(f"  June: peak@{p['avg_peak_hour']:.0f}h  "
          f"+{p['avg_rise_2pm']:.1f}°F after 2pm ({p['pct_rising_2pm']:.0f}%)  "
          f"+{p['avg_rise_4pm']:.1f}°F after 4pm ({p['pct_rising_4pm']:.0f}%)  "
          f"{p['days_analyzed']} days")

def main():
    print("=" * 60)
    print("Building Historical Temperature Database")
    print("Source: Open-Meteo Archive API (5 years)")
    print(f"Cities: {len(CITIES)}")
    print("=" * 60)

    ensure_schema()

    total_days = 0
    for station_code, (lat, lon, city_name) in CITIES.items():
        print(f"\n[{station_code}] {city_name} ({lat}, {lon})")

        # Fetch hourly data
        records = fetch_historical_hourly(lat, lon, start_year=2021)
        if not records:
            print(f"  No data — skipping")
            continue

        # Process daily highs
        utc_offset  = STATION_UTC_OFFSET.get(station_code, -6)
        daily_highs, date_hours = process_daily_highs(records, station_code, utc_offset)
        print(f"  {len(daily_highs)} days of daily highs calculated")

        # Save daily highs
        saved = save_daily_highs(station_code, city_name, daily_highs)
        total_days += saved
        print(f"  {saved} daily highs saved to DB")

        # Calculate and save heating profiles
        profiles = calculate_heating_profiles(date_hours, station_code, city_name)
        save_heating_profiles(station_code, city_name, profiles)
        print_june_summary(station_code, city_name, profiles)

        # Be polite to Open-Meteo
        time.sleep(1)

    print("\n" + "=" * 60)
    print(f"Done! {total_days} total daily highs stored in PostgreSQL")
    print("Tables: historical_highs, heating_profiles_db")
    print("Bot can now query real bias corrections and heating profiles")
    print("=" * 60)

if __name__ == "__main__":
    main()
