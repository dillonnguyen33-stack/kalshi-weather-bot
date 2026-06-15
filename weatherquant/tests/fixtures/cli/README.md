# Vendored NWS CLI parity fixtures

Each `<CITY>.json` carries a **winter** day and a **summer (civil-DST)** day for one of
the 7 in-scope Kalshi cities, used by `tests/test_cli_parity.py` (D-04). A fixture's
hourly `obs` (UTC-timestamped, whole-degree F), when bucketed into
`settlement_window(city, day)`'s half-open `[start_utc, end_utc)` window and reduced
with `max`, must equal that day's `cli_max` (the NWS Daily Climate Report "Maximum"
Observed Value). These fixtures prove the **window math**, not live ingestion (ingestion
is Phase 2). They are deterministic and reproducible via `tests/fixtures/_gen_fixtures.py`.

## Why a "trap" reading per day

Each day also includes two readings **hotter** than `cli_max` placed deliberately
outside the window: one exactly **at** `end_utc` (which is EXCLUSIVE — half-open, D-03)
and one **one hour before** `start_utc` (belongs to the prior LST day). A sign error in
the offset→UTC conversion (Pitfall 3) or an inclusive `end_utc` would wrongly pick up a
trap and the parity test would fail. This is intentional regression armor.

## Source-of-truth (NWS Daily Climate Report)

CLI products are published at:
`https://forecast.weather.gov/product.php?site=<WFO>&product=CLI&issuedby=<CLI3>`
Historical CLI archives are also available via the Iowa Environmental Mesonet (IEM) CLI
product archive. The exact URL per city is stored in each fixture's `source_url` field.

| City | Station | WFO | CLI3 | std_offset | Winter day | Summer day |
|------|---------|-----|------|-----------|------------|------------|
| NYC  | KNYC | OKX | NYC | -5 | 2025-01-15 | 2024-07-15 |
| CHI  | KMDW | LOT | MDW | -6 | 2025-01-15 | 2024-07-15 |
| AUS  | KAUS | EWX | AUS | -6 | 2025-01-15 | 2024-07-15 |
| MIA  | KMIA | MFL | MIA | -5 | 2025-01-15 | 2024-07-15 |
| LAX  | KLAX | LOX | LAX | -8 | 2025-01-15 | 2024-07-15 |
| DEN  | KDEN | BOU | DEN | -7 | 2025-01-15 | 2024-07-15 |
| PHI  | KPHL | PHI | PHL | -5 | 2025-01-15 | 2024-07-15 |

> Stations are the **verified Kalshi settlement stations** (Austin = KAUS, LA = KLAX —
> NOT KATT / KCQT). See `01-RESEARCH.md` § Verified Kalshi Settlement Stations.

## Regenerating

```bash
uv run python tests/fixtures/_gen_fixtures.py
```

The `cli_max` values are representative whole-degree-F daily highs for the named station
on the cited dates; to refresh against the live archived CLI product, update the `DAYS`
table in `_gen_fixtures.py` with the value read from the report's
Temperature → Observed Value → "Maximum" row and rerun.
