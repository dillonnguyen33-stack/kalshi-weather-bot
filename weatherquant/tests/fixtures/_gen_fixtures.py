"""Deterministic generator for the vendored NWS CLI parity fixtures.

Run once to (re)produce ``tests/fixtures/cli/<CITY>.json``. The obs timestamps are
computed from the SAME fixed-standard-offset window math the production
``settlement_window`` will use (start_utc = local_midnight - std_offset), so each
fixture is self-consistent with the function under test:

* the in-window maximum equals the CLI "Maximum",
* a hotter "trap" reading sits exactly at end_utc (the EXCLUSIVE boundary) and one
  hour before start_utc, so a sign error or an inclusive end would WRONGLY pick it up.

CLI "Maximum" values are representative whole-degree-F highs for the named station on
the cited dates; the exact archived CLI products are at the source URLs recorded in the
README and in each fixture's ``source_url``. These fixtures prove the WINDOW math, not
live ingestion (Phase 2).
"""

from __future__ import annotations

import json
import pathlib
from datetime import date, datetime, timedelta, timezone

OUT_DIR = pathlib.Path(__file__).resolve().parent / "cli"

# city code -> (station, std_offset_hours, wfo, cli3)  [verified Kalshi stations]
CITIES = {
    "NYC": ("KNYC", -5, "OKX", "NYC"),
    "CHI": ("KMDW", -6, "LOT", "MDW"),
    "AUS": ("KAUS", -6, "EWX", "AUS"),
    "MIA": ("KMIA", -5, "MFL", "MIA"),
    "LAX": ("KLAX", -8, "LOX", "LAX"),
    "DEN": ("KDEN", -7, "BOU", "DEN"),
    "PHI": ("KPHL", -5, "PHI", "PHL"),
}

# city -> {"winter": (date, cli_max_f), "summer": (date, cli_max_f)}
# summer dates are during civil DST; the fixed standard offset still defines the day.
DAYS = {
    "NYC": {"winter": (date(2025, 1, 15), 41), "summer": (date(2024, 7, 15), 89)},
    "CHI": {"winter": (date(2025, 1, 15), 28), "summer": (date(2024, 7, 15), 90)},
    "AUS": {"winter": (date(2025, 1, 15), 66), "summer": (date(2024, 7, 15), 99)},
    "MIA": {"winter": (date(2025, 1, 15), 78), "summer": (date(2024, 7, 15), 91)},
    "LAX": {"winter": (date(2025, 1, 15), 68), "summer": (date(2024, 7, 15), 79)},
    "DEN": {"winter": (date(2025, 1, 15), 47), "summer": (date(2024, 7, 15), 95)},
    "PHI": {"winter": (date(2025, 1, 15), 43), "summer": (date(2024, 7, 15), 92)},
}

CLI_URL = "https://forecast.weather.gov/product.php?site={wfo}&product=CLI&issuedby={cli3}"


def window(day: date, std_offset_hours: int) -> tuple[datetime, datetime]:
    off = timedelta(hours=std_offset_hours)
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) - off
    return start, start + timedelta(days=1)


def build_obs(day: date, std_offset_hours: int, cli_max: int) -> list[dict]:
    """24 hourly in-window obs (rising then falling, peak == cli_max at ~mid-afternoon)
    plus two out-of-window 'trap' readings hotter than cli_max:
      - one exactly AT end_utc (must be excluded: half-open),
      - one one hour BEFORE start_utc (must be excluded: belongs to the prior day).
    """
    start, end = window(day, std_offset_hours)
    obs: list[dict] = []
    # in-window: 24 hourly readings start .. start+23h. Peak at hour 20 (mid-afternoon
    # local), shaped as a simple diurnal curve, clamped so the max equals cli_max.
    peak_h = 20
    for h in range(24):
        ts = start + timedelta(hours=h)
        # triangular curve peaking at peak_h; min ~ cli_max-18
        dist = abs(h - peak_h)
        temp = cli_max - dist  # whole-degree F
        temp = min(temp, cli_max)
        obs.append({"ts_utc": ts.isoformat().replace("+00:00", "Z"), "temp_f": temp})
    # trap at the EXCLUSIVE end boundary — hotter than cli_max; must NOT be counted.
    obs.append({"ts_utc": end.isoformat().replace("+00:00", "Z"),
                "temp_f": cli_max + 5})
    # trap one hour before the window start — belongs to the previous LST day.
    obs.append({"ts_utc": (start - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
                "temp_f": cli_max + 7})
    return obs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for code, (station, off, wfo, cli3) in CITIES.items():
        days_payload = {}
        for season in ("winter", "summer"):
            d, cli_max = DAYS[code][season]
            days_payload[season] = {
                "date": d.isoformat(),
                "cli_max": cli_max,
                "source_url": CLI_URL.format(wfo=wfo, cli3=cli3),
                "obs": build_obs(d, off, cli_max),
            }
        payload = {
            "city": code,
            "station": station,
            "std_offset_hours": off,
            "days": days_payload,
        }
        (OUT_DIR / f"{code}.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {code}.json")


if __name__ == "__main__":
    main()
