<!-- GSD:project-start source:PROJECT.md -->

## Project

**Weatherquant**

Weatherquant is a probabilistic forecasting system that prices bets on Kalshi daily-high-temperature markets. It pulls raw NOAA weather-model data, fits a statistically calibrated probability distribution per model/city/lead-time/month, blends the models via Vincentization, and sizes positions with a fee-aware fractional Kelly criterion. It is the successor to the v3 Discord-alert bot, fixing v3's two core failures: uncalibrated probabilities and never actually placing orders.

**Core Value:** Produce **statistically calibrated** probabilities for Kalshi temperature buckets — a 70% prediction must resolve YES ~70% of the time — and prove that edge against v3 before any real money is risked.

### Constraints

- **Tech stack**: Pure NumPy for calibration — no scipy/sklearn — to keep the deploy lightweight.
- **Correctness**: Temperature settlement window must use Local Standard Time (no DST), matching Kalshi resolution exactly.
- **Statistical**: Blending must use quantile-averaging Vincentization, not linear mixture (a linear mix of calibrated forecasts is overdispersed).
- **Risk**: No live orders until Gate 1 proves edge with confidence intervals excluding zero; single-position cap of 2–5% of bankroll.
- **Budget**: $500 bankroll; daily running cost kept low (free tiers ~$3–8/day). Commercial Open-Meteo license (~$400/mo) required before live trading use.
- **Data**: NOAA GRIB fetched via byte-range requests to keep S3 egress negligible.

<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->

## Technology Stack

## The one hard constraint that shapes the whole stack

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **Python** | 3.12 (3.11+ min) | Runtime | 3.12 is the sweet spot: every dep below supports it; the official Kalshi SDK and `kalshi-sdk` need 3.12+; avoid 3.13/3.14 edge cases with compiled wheels (eccodes, asyncpg). |
| **NumPy** | 2.4.6 | Calibration core (EMOS/NGR, CRPS, Vincentization, EV, Kelly) | The mandated and *sufficient* engine. Closed-form Gaussian-CRPS gradient → hand-rolled gradient descent (or Adam) in ~50 lines. No scipy needed. NumPy 2.x is stable and ABI-compatible across the rest of the stack. |
| **eccodes** (C lib) + **cfgrib** | eccodes 2.47.x / cfgrib 0.9.15.1 | Decode GRIB2 temperature fields into arrays | `cfgrib` is the de-facto standard GRIB2→xarray bridge (ECMWF-maintained, same org as eccodes). Unavoidable: GRIB2 is a packed binary format; no pure-NumPy decoder exists. Pin eccodes via conda-forge or the `eccodes` wheel to avoid system-library hell. |
| **boto3** + **botocore UNSIGNED** | boto3 1.43.x | Anonymous byte-range reads from NOAA public S3 | NOAA buckets (`noaa-hrrr-pds`, `noaa-gfs-bdp-pds`, `noaa-gefs-pds`, `noaa-nbm-grib2-pds`) are free public Open-Data buckets. Use `Config(signature_version=UNSIGNED)` + `Range:` header on `get_object` to pull only the bytes for `TMP:2 m` (≈1 MB vs ≈700 MB full file) → egress stays negligible, matching the PROJECT.md cost target. |
| **psycopg** (v3) | 3.3.4 | Postgres driver | Modern successor to psycopg2. Sync + async in one library, server-side binding, native `COPY`, good typing. Pair with `psycopg[binary]` wheel so no local libpq build. |
| **anthropic** | 0.109.x | Claude Haiku — classify NWS Area Forecast Discussion text | Official SDK; built-in retries, streaming, structured/tool use for clean JSON extraction of "forecaster disagreement" flags. Use `claude-haiku` tier for the per-city AFD classification (cheap, matches the $3–8/day budget). |
| **Kalshi official SDK** (`kalshi-python`) | 2.1.4 | Kalshi REST (markets, orderbook snapshot, paper-account state) | Official (maintained by Kalshi Support; repo `Kalshi/exchange-infra`). Handles RSA-PSS signing internally via `config.private_key_pem`. REST-only — see WebSocket note below. |

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| **xarray** | 2026.4.0 | Labeled n-d arrays returned by cfgrib | Convenient for selecting `(lat, lon, lead_time)` from decoded GRIB. Keep it at the *I/O edge* only — convert to plain `np.ndarray` before the calibration core so the pure-NumPy boundary stays clean. |
| **websockets** | 16.0 | Kalshi live orderbook WS for paper-fill simulation | Pure-asyncio, modern, well-maintained. The official Kalshi SDK does **not** ship a WS client, so you implement the orderbook feed here: sign the WS handshake with the same RSA-PSS scheme, subscribe to `orderbook_delta`, maintain the book for queue-position/partial-fill simulation. Prefer over `websocket-client` (1.9.0, thread-based, dated API). |
| **cryptography** | 49.0.0 | RSA-PSS request signing (if/where you sign manually) | The SDK signs REST for you, but the **WebSocket** handshake and any custom calls need manual signing: `RSA-SHA256` with `PSS` padding over `timestamp + method + path`. `cryptography` is the standard, audited primitive — do not hand-roll. |
| **httpx** | 0.28.1 | Async HTTP for non-SDK calls (Open-Meteo, NWS API, METAR/ASOS) | One async client for all the supplementary REST sources in PROJECT.md. HTTP/2, timeouts, retry-friendly. |
| **APScheduler** | 3.11.2 | Time-triggered ingestion (model runs land on fixed UTC cycles) | In-process cron-like scheduler. NWP cycles are deterministic (HRRR hourly, GFS/GEFS 00/06/12/18Z, NBM hourly) → schedule fetch jobs per cycle. Lightweight; no external broker. Good enough for a single-node paper-trading system. |
| **SQLAlchemy** (Core) + **Alembic** | 2.0.50 / 1.18.4 | Schema definition + migrations | Use SQLAlchemy **Core** (not the full ORM) to declare tables for forecasts/calibration-params/fills, and Alembic for versioned migrations. Core keeps you close to SQL while giving Alembic autogeneration. (If you prefer raw SQL, Alembic still works standalone.) |
| **pydantic-settings** | 2.14.1 | Typed config + env/secret loading | Single typed `Settings` object: `KALSHI_KEY_ID`, `KALSHI_PRIVATE_KEY_PATH`, `ANTHROPIC_API_KEY`, `DATABASE_URL`, `EXECUTION_MODE`. Validates at startup, fails loud on missing secrets. |
| **python-dotenv** | 1.2.2 | Local `.env` loading | Dev-only convenience; pairs with pydantic-settings. Never commit the `.env`. |
| **timezonefinder** | 8.2.4 | Map city lat/lon → IANA tz for the LST settlement window | Directly serves the v3 bug fix: Kalshi settles daily-high on **Local Standard Time (no DST)**. Resolve each city's base UTC offset and subtract DST manually so the settlement window is correct year-round. |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| **uv** | Dependency + venv manager | Fast, lockfile-based (`uv.lock`), reproducible. Far better than pip+venv for pinning the compiled-wheel matrix (eccodes/cfgrib/asyncpg). |
| **ruff** | Lint + format | One tool replaces flake8/isort/black. Fast, zero-config-friendly. |
| **pytest** + **pytest-asyncio** | Tests | Needed for TDD gates; async tests for the WS/orderbook simulator. Property tests on the CRPS gradient (finite-difference check) catch calibration-math bugs early. |
| **mypy** (or ruff's type checks) | Static typing | Worthwhile on the money path (EV, Kelly, fills) where unit/sign errors are expensive. |

## Installation

# Use uv for reproducible, lockfile-based installs

# --- Calibration core (the ONLY thing the constraint governs) ---

# --- GRIB ingestion (compiled libs are unavoidable here) ---

# --- Kalshi (REST official SDK + manual WS) ---

# --- Persistence ---

# --- LLM + supplementary data ---

# --- Orchestration / config / tz ---

# --- Dev ---

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `boto3` UNSIGNED + manual `Range` reads | **Herbie** (`herbie-data` 2026.3.0) | Herbie is excellent and *purpose-built* for NWP subset downloads from NOAA S3 (parses `.idx`, byte-range cURL, knows HRRR/GFS/GEFS/NBM layouts). Use it if you want the `.idx`-parsing wheel pre-built rather than writing your own offset logic. Trade-off: it's a heavier dependency tree and another abstraction over the data; raw boto3 + your own thin `.idx` parser gives full control over exactly which bytes you fetch and keeps the ingestion code auditable. **Reasonable to adopt Herbie for the `.idx`→byte-range step specifically** and still decode with cfgrib. |
| `cfgrib` | **pygrib** (2.1.8) | pygrib is fine and sometimes simpler for "give me one field"; it also binds eccodes/grib-api. cfgrib wins for xarray integration and ECMWF maintenance. Either satisfies the (unavoidable) compiled-GRIB requirement. |
| `kalshi-python` (official) | **`kalshi-sdk`** (TexasCoding, 4.1.0) | Community SDK with more ergonomic typing and async; needs 3.12+. Use if the official SDK's coverage/feedback loop frustrates you. The official one is the safer default for auth correctness. |
| `websockets` (asyncio) | **`websocket-client`** (1.9.0) | Only if you're committed to a thread-based, non-asyncio architecture. For a paper-fill simulator that also schedules async ingestion, stay all-asyncio with `websockets`. |
| `psycopg` v3 | **asyncpg** (0.31.0) | asyncpg is faster for high-throughput async inserts. For paper-trading volumes, psycopg3's simpler sync+async story and `COPY` support are plenty; reach for asyncpg only if fill-logging becomes a hot path. |
| `APScheduler` | **cron / systemd timers** | If you'd rather keep scheduling outside the process (one-shot scripts triggered by OS cron), that's leaner and crash-isolated. APScheduler wins for keeping everything in one observable Python process with shared state. Avoid Airflow/Prefect — massive overkill for a single-node bot. |
| SQLAlchemy Core + Alembic | **raw SQL + yoyo/dbmate** | If you dislike SQLAlchemy, a plain migration runner over raw `.sql` files is perfectly valid. Core+Alembic chosen for autogeneration and one ecosystem. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **scipy** (anywhere in calibration) | Hard PROJECT.md constraint (deploy weight). And unnecessary: Gaussian CRPS has a closed-form value *and gradient*, so `scipy.optimize.minimize` / `scipy.stats.norm` are not needed. | Pure NumPy: implement `norm.cdf/pdf` via `math.erf`/`np.special`-free closed forms, and minimize CRPS with hand-written gradient descent / Adam. |
| **sklearn** | Hard constraint; EMOS/NGR is a tiny bespoke regression, not a sklearn estimator. Pulling sklearn drags in scipy too. | Pure NumPy least-squares init (`np.linalg.lstsq`) for warm-start, then NumPy gradient descent on the CRPS objective. |
| **scipy.stats.norm in CRPS** (subtle) | Easy accidental violation — `scipy.stats` is the "natural" place people reach for the normal CDF. | NumPy-only normal CDF/PDF. The CRPS closed form needs only Φ, φ — both expressible without scipy. |
| **Linear mixture of calibrated forecasts** | PROJECT.md statistical constraint: a linear mix of calibrated distributions is **overdispersed**. | **Vincentization** = average the *quantiles* (pure NumPy: `np.quantile` per model, then mean across models). |
| **UTC settlement window** | The exact v3 bug. Kalshi settles daily-high on **Local Standard Time, no DST** → a UTC window shifts the day and flips which trades look profitable. | Resolve city tz with `timezonefinder`, use the **standard** offset (strip DST) to define the daily window. |
| **Downloading full GRIB2 files** | ≈700 MB/HRRR run × 4 models × many cycles = real egress + storage cost, violating the cost target. | `.idx`-driven **byte-range** `get_object(Range=...)` for the `TMP:2 m` message only (≈1 MB). |
| **Airflow / Prefect / Dagster** | Heavyweight orchestration for a single-node paper bot; ops burden >> value here. | APScheduler in-process, or OS cron. |
| **Storing the RSA private key in `.env`/repo/ConfigMap** | Loss = irrecoverable (Kalshi doesn't store it); leak = account-level trading access. | Local: file path outside the repo (`KALSHI_PRIVATE_KEY_PATH`), referenced via pydantic-settings. Future live (Gate 2): AWS Secrets Manager / Vault. Never commit. |
| **psycopg2** | Legacy; no native async, slower dev. | `psycopg` v3. |
| **requests** for hot paths | Sync-only; clashes with the asyncio WS/scheduler design. | `httpx` (async) for supplementary sources; SDK clients for Kalshi/Anthropic. |

## Stack Patterns by Variant

- Use **Herbie** (`herbie-data`) for the S3 discovery + `.idx` byte-range subset step, then hand its GRIB output to cfgrib.
- Because Herbie already encodes the per-model S3 path templates and index formats for HRRR/GFS/GEFS/NBM — less bespoke offset math to maintain.
- Use **raw boto3 UNSIGNED** + a ~30-line `.idx` parser + cfgrib.
- Because you control exactly which bytes are fetched and the ingestion path is fully auditable (matters when egress cost and correctness are both graded).
- Swap `psycopg` for **asyncpg** on the write path only.
- Because async batched inserts under live order flow benefit from asyncpg's lower per-statement overhead.
- Batch AFD calls and use Anthropic **structured/tool-use** output for deterministic JSON.
- Because per-city-per-cycle calls add up; batching + structured output cuts cost and parsing fragility, staying inside the $3–8/day target.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `numpy@2.4.x` | `xarray@2026.4`, `cfgrib@0.9.15` | NumPy 2.x ABI is the baseline for current xarray/cfgrib wheels — no NumPy-1.x pinning needed. |
| `cfgrib@0.9.15.1` | `eccodes@2.47.x` (C lib) | cfgrib binds the eccodes C library; the Python wheel ≠ the binary. Install the eccodes binary (conda-forge or OS pkg) if the wheel can't locate it. **#1 install pitfall.** |
| `psycopg@3.3.x` | `sqlalchemy@2.0.x`, `alembic@1.18.x` | SQLAlchemy 2.0 supports psycopg3 via the `postgresql+psycopg://` dialect. Use that URL scheme, not the psycopg2 default. |
| `kalshi-python@2.1.4` | Python ≥3.9 | REST only. WebSocket handled separately via `websockets`+`cryptography`. Confirm the SDK is initialized with `private_key_pem`. |
| `kalshi-sdk@4.1.0` (if chosen instead) | Python ≥3.12 | Forces 3.12+ — fine given the recommended runtime, but a constraint to note. |
| `websockets@16.0` | Python ≥3.10, asyncio | Modern asyncio API (`websockets.connect` as async ctx mgr); ignore pre-12.0 tutorials. |
| `apscheduler@3.11.x` | asyncio | Use `AsyncIOScheduler` to share the event loop with the WS feed and httpx clients. (Note: APScheduler 4.x is a different, in-flux API — stay on 3.11.x.) |

## Sources

- PyPI JSON API (live, 2026-06-15) — verified current versions: numpy 2.4.6, cfgrib 0.9.15.1, xarray 2026.4.0, eccodes 2.47.0, boto3 1.43.29, psycopg 3.3.4, kalshi-python 2.1.4, kalshi-sdk 4.1.0, anthropic 0.109.1, websockets 16.0, cryptography 49.0.0, httpx 0.28.1, apscheduler 3.11.2, sqlalchemy 2.0.50, alembic 1.18.4, pydantic-settings 2.14.1, python-dotenv 1.2.2, timezonefinder 8.2.4, herbie-data 2026.3.0, pygrib 2.1.8, asyncpg 0.31.0 — **HIGH**
- docs.kalshi.com — RSA-PSS (RSA-SHA256, PSS padding) over `timestamp+method+path`; official SDK signs internally — **HIGH**
- pypi.org/project/kalshi-python — official, maintained by Kalshi Support (repo `Kalshi/exchange-infra`); REST-only, `config.private_key_pem` — **HIGH**
- registry.opendata.aws (NOAA GFS/GEFS/HRRR/NBM) + AWS CLI `--no-sign-request` guidance — free public buckets, anonymous access — **HIGH**
- Brian Blaylock HRRR script tips + Herbie GRIB2 docs (herbie.readthedocs.io) — `.idx` byte-range subsetting (TMP:2 m ≈1 MB of a ≈700 MB file); NCEP wgrib2-style index — **HIGH**
- github.com/ecmwf/cfgrib — cfgrib↔eccodes relationship, xarray mapping — **HIGH**
- PROJECT.md (this repo) — no-scipy/no-sklearn constraint, LST settlement, Vincentization-not-linear-mix, byte-range egress target — **authoritative project source**

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

- Decision rationale is single-sourced in `docs/DECISIONS.md`; modules cite a decision by a
  **subtree-local** id (the same `D-01` means different things in core vs `calibrate/` vs `price/`)
  plus a one-line WHY. Docstrings carry the WHY, not WHAT-narration.
- Guarded invariants (enforced by `tests/`): pure-NumPy in `calibrate/`+`price/` (no scipy/sklearn),
  no runtime DST tooling, no `market` import into `price/`, paper-only (no live orders).
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

One-way dependency spine, no import cycles: `time`/`registry` → `ingest` (one orchestrator code
path, live==backfill) → `calibrate` (EMOS/NGR fits) → `price` (`blend`/`fee`/`buckets` CDF +
`ticker` parsing) ; `market/` (paper book/CLV) and the `cli/` package (one submodule per
subcommand) sit at the edges. `db/` is the shared audited-writer ledger.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
