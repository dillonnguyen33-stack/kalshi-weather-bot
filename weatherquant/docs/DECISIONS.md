# Weatherquant decisions

Decision rationale extracted from inline docstrings, so source modules can cite a
decision by id and a one-line WHY instead of carrying the full narrative.

**Decision ids are subtree-local.** The same id means different things in different
phases: `D-01` is the no-DST settlement window in the core/ingest tree, the
label-generic stratifier in `calibrate/`, and Gaussian Vincentization in `price/`.
`RESEARCH Pitfall N` is likewise per-phase. Read each id within its own section; the
citing file's location disambiguates which scheme applies.

---

## Core / time / registry / ingest / market / db / cli

### D-01 — Fixed-offset LST settlement, no runtime DST
Kalshi settles daily-high on the NWS Daily Climate Report, whose climatological day is
midnight-to-midnight Local **Standard** Time, ignoring DST year-round. The v3 bug used
DST-aware tz math, shifting the day an hour during DST. The window is computed purely
from `city.std_offset_hours`; the runtime path imports neither `zoneinfo` nor
`timezonefinder`.

### D-02 — Offset derivation verified in tests only
The fixed `std_offset_hours` int is the single runtime source of truth. Cross-checking
it against `ZoneInfo` on a January (standard-time) date lives only in tests, so no tz
tooling reaches the runtime path (`test_no_runtime_dst.py` guards this).

### D-03 — Half-open settlement window
`SettlementWindow` is `[start_utc, end_utc)` — `end_utc` is exclusive and equals the
next day's `start_utc`, always exactly 24h (true precisely because there is no DST), so
boundary observations are never double-counted.

### D-04 — EV + minimum-stake Kelly gate
Place one sized paper position per `(market, side)`, held to settlement, only when EV is
positive and the Kelly stake ≥ `PAPER_MIN_STAKE_FRACTION` (1e-4). A sub-minimum stake is
"no edge worth the spread", not a churned micro-order.

### D-05 — Typed registry, not JSON/TOML
The city registry is a frozen `City` dataclass + `CITIES` dict — typed, importable,
unit-testable, no runtime file I/O for a tiny static dataset.

### D-06 — Exactly the 7 in-scope cities
Only the 7 verified Kalshi daily-high cities; adding a city is just another dict entry.

### D-07 — Station elevation required, in meters
Coordinates/elevation are the CLI station's own (the GRIB point is taken at the station,
not the city centroid). Elevation is required and stored in meters (SI).

### D-08 — EV and Kelly share one shrunk belief
`bucket_ev` shrinks the model prob toward the market mid (`p_used`); Kelly sizes on that
**same** shrunk belief so the printed edge and the stake agree in sign near the boundary.

### D-09 — `available_at` never the wall clock for history
`available_at` = `cycle_init + PUBLISH_LATENCY` in backfill, decode-completion `now(UTC)`
in live — never `now()` for a historical row. The no-look-ahead invariant Phase 6 depends
on; a dishonest stamp raises `AvailabilityError`.

### D-10 — Skip-before-insert idempotency
Re-running the same range/cycle is a no-op: each duplicate is skipped before insert and
returns 0 rows.

### D-11 — Graceful degradation (absence = absence)
Each source is wrapped in its own try/except; a late/missing cycle or fetch/decode error
logs a structured `(model, city, cycle, reason)` fallback and ingestion proceeds with the
other sources. Nothing is synthesized, neighbor-filled, or fabricated. Correctness alarms
(`CorrectnessError` subclasses + `AssertionError`) are excluded by type and fail loud.

### D-12 — Provider namespacing
The same underlying model from two providers (NOAA `hrrr` vs `wethr:hrrr`) stays two
distinct blend inputs — never deduped or merged.

### D-13 — AFD pre-filter + forced tool-use
A v3 keyword pre-filter gates any paid Anthropic call (budget). When it fires, a forced
`record_afd_signal` tool with a strict `input_schema` returns a guaranteed-shape dict
(vs v3's fragile JSON-text parse) and contains injected free-text (ASVS V5). Key unset →
structured skip, no-signal. The signal is a soft sizing modifier, never a hard trade gate.

### D-14 — Off-loop decode
The sync Herbie+cfgrib GRIB decode and the blocking Anthropic classify both run in a
thread executor so they never block the async loop shared by the HTTP sources.

### D-15 — One ingestion code path
The live scheduler and the backfill CLI both call `orchestrator.ingest_cycle`; the only
difference is `mode` (the `available_at` stamp). No second, drifting path the backtest
could diverge from.

### D-16 — LST `target_date`, no hand-rolled UTC day
A forecast's `target_date` is resolved by walking the city's half-open
`settlement_window` for the valid instant — never a hand-rolled UTC day (the v3
anti-pattern). The obs path labels against the same window, so forecasts and truth join.
Closed by the live-gated KXHIGH money-path cross-check (bucket-edge half-degree, fee
0.07, ticker-suffix → registry).

### Pitfalls (phase-01)
- **Pitfall 1** — A fixed integer standard offset is *more* correct here than a DST-aware
  `ZoneInfo` conversion (the no-DST inversion).
- **Pitfall 2** — LA settles on KLAX (airport / `KXHIGHLAX`), not KCQT/Downtown; Austin on
  KAUS, not Camp Mabry. Verified against Kalshi contract terms — do not "fix" to v3 values.
- **Pitfall 3** — Local-standard midnight in UTC is `local_midnight − offset`; US offsets
  are negative, so `start_utc.hour == (−std_offset_hours) % 24`.

---

## Calibrate

### D-01 — Model-label-generic stratification
The read → aggregate → fit path keys on whatever `model` string is present (the four NOAA
models and the supplementary blend inputs alike); no per-model `if model ==` branch.

### D-02 — NGR parameterization + masked CRPS gradient
`mu = a + b·m`, `var = max(σ_floor², c² + d²·s2)`. Squaring `c`,`d` makes variance
non-negative by construction (no constrained optimization). The clamp is non-differentiable
at the kink: when the floor is active, `dσ/dc = dσ/dd = 0` must be masked, or the optimizer
takes a phantom step it can't escape.

### D-03 — Single K→°F seam
`kelvin_to_fahrenheit` is the one named K→°F conversion on the calibration path (mirroring
`obs.celsius_to_fahrenheit`); no other `calibrate/` module inlines the `9/5` / `459.67`.

### D-04 — Closed-form Gaussian CRPS
The most safety-critical artifact: value and gradient in exact closed form (Gneiting et al.
2005), never numerical integration — a wrong-but-plausible CRPS still "optimizes".

### D-05 — Finite-difference gradient check
The structural guard on the CRPS gradient is the finite-difference test, which is why the
gradient is hand-derived in closed form.

### D-07 — `month` derived from `target_date`
`month` (part of the natural key) is derived from `target_date` after the full-key read.

### D-08 — Pooling ladder + shrinkage
A data-starved fine stratum coarsens its most data-starved axis first
(`month → season → adjacent leads`) and shrinks toward the pooled parent, instead of a
degenerate over-confident fit that would blow up Kelly sizing.

### D-09 — Additive σ-floor
The additive `sigma_floor` clamp blocks the degenerate over-confidence a near-zero σ would
otherwise feed into Kelly sizing.

### D-10 — Temporal OOS split
`temporal_split` orders by `target_date` into an earlier train slice and a strictly-later
OOS slice. A shuffled split leaks future info (look-ahead) and is the structural
no-look-ahead guard for the Phase-3 sanity check.

### D-11 — Raw-ensemble baseline
The without-EMOS baseline is `(mu = m, sigma = sqrt(s2))`. A genuinely deterministic
stratum (`s2 == 0`) falls back to the train-slice residual std — never a degenerate `σ=0` —
computed from train only.

### D-12 — Anti-p-hacking scope
Fit hyperparameters (`N_MIN`, `KAPPA`, `SIGMA_FLOOR_F`, Adam settings) are fixed research
defaults, not tuned against this OOS slice, which stays disjoint from Phase 6's Gate-1 set.

### D-13 — Append-only, point-in-time persistence
A refit is a fresh insert (the append-only trigger rejects mutation); `latest()` returns
the current params. `available_at` is a parameter (training-run completion instant), never
`now()`; `trained_through` is the data cutoff so Phase 6 can re-derive any historical fit.

### D-14 — Shared params→Gaussian link
`predict()` is the single source of truth for `params → (mu, sigma)`, reused verbatim by
pricing — no divergent re-implementation between the fit and the downstream price.

### Pitfalls (phase-03)
- **Pitfall 1** — Mask the variance-param gradient to 0 when the σ-floor clamp is active.
- **Pitfall 2** — A deterministic single-member model has `s2 == 0` ⇒ `d` inactive
  (gradient identically 0): expected, not a bug.
- **Pitfall 3** — Read with the full natural key `(city, target_date, model, lead, member)`;
  an under-specified key collapses distinct ensemble members into one wrong "truth"
  (`latest()` rejects it with `ValueError`).
- **Pitfall 4** — Temporal split, never shuffled (look-ahead).

---

## Price

### D-01 — Gaussian Vincentization closed form
Quantile-average the calibrated per-model Gaussians into one `N(Σwᵢμᵢ, (Σwᵢσᵢ)²)`:
`σ_blend` is the weighted **mean** of the std-devs, NOT `sqrt(Σwᵢσᵢ²)` and NOT the
overdispersed linear-mixture variance. Makes `σ_blend ≤ max(σᵢ)` true by construction.

### D-02 — Inverse-CRPS weights with a floor
Weights are normalized inverse OOS-CRPS (lower CRPS ⇒ higher weight), with `CRPS_EPS`
guarding a near-zero CRPS and a `W_MIN` floor so no model fully dominates or drops out.

### D-03 — NULL-`crps_oos` fallback
A NULL `crps_oos` (pure-pooled fit) falls back to its pooled-parent CRPS or equal weight; a
missing model drops out and survivors renormalize. `crps_oos` is a relative cross-model
signal only, never an absolute quality measure.

### D-07 — Exact integer-cent fee
`fee = ceil(round(FEE_COEFF·n·p·(1−p)·100, 9)) / 100` — rounded up once per order (never
per-contract, never to-nearest). The round-to-9dp absorbs IEEE-754 noise before the ceil.
`FEE_COEFF = 0.07` is MEDIUM confidence (some product lines use 0.035 — verify KXHIGH live).

### D-09 — Maker fee off the sizing path
The maker fee is parameterized off the taker fee (default 0.25×, "verify per market").
Gate-1 sizes on the taker fee only — the maker helper is exposed but never on the path.

### Pitfalls (phase-04)
- **Pitfall 2** — A linear mixture of calibrated forecasts is overdispersed → Vincentize.
- **Pitfall 3** — Fee is ceiled once per order; round before the ceil to erase IEEE-754 noise.
- **Pitfall 4** — NULL-`crps_oos` weight fallback (see D-03).
- **Pitfall 5** — `crps_oos` is a relative cross-model signal, not an absolute quality measure.

---

## Work rationale (WR-XX)

- **WR-01** — Thread `mode` through `available_at` (never hardcode `"live"`) so the
  live/backfill seam stays genuinely single.
- **WR-02** — Live-only sources (nws/openmeteo/wethr) are not backfilled — they have no
  point-in-time historical archive, so backfill skips them with a structured log
  (absence = absence) rather than stamping today's forecast onto a past date.
- **WR-03** — The blocking AFD classify runs off-loop in a thread executor.
- **WR-04** — Fail loud on an impossible `target_date`: raise `TargetDateError` (a
  `CorrectnessError`, not a bare `ValueError`, so the orchestrator's try doesn't swallow it)
  rather than substitute a hand-rolled UTC date.
- **WR-05** — Re-raise correctness alarms by base class (`CorrectnessError`). The earlier
  gap caught only `WriteIntegrityError`/`AssertionError`, silently swallowing bare-ValueError
  alarms.
- **WR-06** — Guard rowcount with an explicit `raise`, not a bare `assert`, so it survives
  `python -O` / `PYTHONOPTIMIZE`. (In `cli`, WR-06 is the single host-resolution seam.)
- **WR-07** — Aggregate size per price before taking the best level from the book.

## Ingest / implementation notes (IN-XX)

- **IN-01** — CLV closing-window axis contract: the closing window selects on `available_at`
  (the snapshot event time), consistently across `clv` and `cli`.
- **IN-02** — Calibrate warm-start / shrinkage target: each season parent is fit on all of
  the season's pairs; the pooling ladder assembles strata programmatically rather than
  returning NaN params.
- **IN-03** — Top-of-book reflection lives in one place (`reflect.py`): `yes_ask =
  100 − best_no_bid`; never read a native ask (there is none) or re-derive the reflection.
