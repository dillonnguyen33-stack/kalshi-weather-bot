"""Shared pytest fixtures for the weatherquant test suite (plan 01-01 scaffold).

Two fixtures defined here:

* ``pg_engine`` — a SQLAlchemy ``Engine`` built from ``DATABASE_URL`` for the ledger
  integration tests (SYS-01). The URL is loaded from a local, git-ignored ``.env`` via
  python-dotenv so the suite runs under ``uv run pytest`` without the variable being
  pre-exported in the shell. If ``DATABASE_URL`` is genuinely unset, integration tests
  ``pytest.skip`` cleanly so the fast (no-DB) subset stays green. The scheme must be
  ``postgresql+psycopg://`` (psycopg v3 — D-09 / Pitfall 4); a wrong scheme fails loud.

* ``cli_fixture`` — loads the vendored NWS CLI parity fixtures (winter + summer obs and
  the CLI "Maximum" per city) used by ``test_cli_parity.py`` (D-04). Pure data; no
  network access, no live ingestion (ingestion is Phase 2).

NOTE: this module deliberately imports ``zoneinfo`` / ``timezonefinder`` nowhere on the
runtime path — those belong only in the test modules that derive/cross-check offsets
(D-02). conftest stays import-light.
"""

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass

import numpy as np
import pytest

# Load DATABASE_URL (and any other vars) from a local .env so `uv run pytest` works
# without the operator pre-exporting the variable. dotenv does not override already-set
# environment variables, so an explicit shell export still wins.
try:
    from dotenv import load_dotenv

    # Resolve the package-root .env (weatherquant/.env), regardless of pytest's cwd.
    _ENV_PATH = pathlib.Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(_ENV_PATH)
    load_dotenv()  # also pick up a CWD .env if present
except ImportError:  # python-dotenv is a declared dep; guard for minimal envs only.
    pass


FIXTURE_DIR = pathlib.Path(__file__).resolve().parent / "fixtures" / "cli"

# Vendored byte-range GRIB2 subsets (HRRR Lambert, GFS lat/lon, GEFS member p01) live
# directly under tests/fixtures/ and decode OFFLINE with cfgrib — no network, no Herbie
# in the test path (RESEARCH § Validation: vendor a real sample, not a mock). The two
# AFD pre-filter text fixtures live here too.
GRIB_FIXTURE_DIR = pathlib.Path(__file__).resolve().parent / "fixtures"


def _database_url() -> str | None:
    """Return DATABASE_URL from the environment, or None if genuinely unset/blank."""
    url = os.environ.get("DATABASE_URL")
    if url is not None and url.strip() == "":
        return None
    return url


@pytest.fixture(scope="session")
def pg_engine():
    """Yield a SQLAlchemy Engine bound to the test Postgres (DATABASE_URL).

    Skips cleanly (not errors) when DATABASE_URL is unset, so the fast subset stays green
    on machines without a database. When set, the engine is obtained via the production
    ``weatherquant.db.engine.get_engine()`` — so the suite exercises the real engine
    (validated psycopg3 scheme, ``hide_parameters``, and ``preserve_rowcount`` so INSERT
    rowcount==1 holds, D-11) rather than an ad-hoc one. A trivial ``SELECT 1``
    connectivity check runs before the integration tests; the schema is built from the
    Core metadata (imported lazily).
    """
    url = _database_url()
    if url is None:
        pytest.skip(
            "DATABASE_URL unset — skipping ledger integration tests "
            "(set DATABASE_URL=postgresql+psycopg://... to enable)."
        )

    # Obtain the PRODUCTION engine. get_settings() inside get_engine() runs the same
    # exact-match psycopg3 scheme validator (Pitfall 4 / D-09); a bad scheme raises here.
    import sqlalchemy as sa

    from weatherquant.db.engine import get_engine

    try:
        engine = get_engine()
    except ValueError as exc:
        pytest.fail(str(exc))

    # Connectivity check — proves Postgres is reachable before integration tests run.
    with engine.connect() as conn:
        conn.execute(sa.text("SELECT 1"))

    # Build the schema from the Core metadata. Imported lazily so the no-DB subset never
    # pays the models import — this fixture only runs when DATABASE_URL is set.
    from weatherquant.db.models import metadata

    # Rebuild from scratch so schema/DDL changes (e.g. new triggers) always take effect:
    # create_all is a no-op on pre-existing tables and would NOT re-run their after_create
    # trigger DDL, leaving a stale schema from a prior run. drop_all first guarantees the
    # current append-only guards (UPDATE/DELETE + TRUNCATE) are installed every session.
    metadata.drop_all(engine)
    metadata.create_all(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def pg_conn(pg_engine):
    """Yield a Connection inside a transaction that is rolled back after each test.

    The ledger is append-only (UPDATE/DELETE/TRUNCATE all raise), so committed test rows
    can never be cleaned up — left uncontrolled they accumulate across runs on the
    session-scoped engine and make row-counting assertions flaky. The guard triggers are
    BEFORE UPDATE/DELETE/TRUNCATE only, so uncommitted INSERTs roll back cleanly here,
    isolating every test. Reads issued on this same Connection see the in-transaction
    rows (pass it straight to ``queries.latest(pg_conn, ...)``).
    """
    conn = pg_engine.connect()
    txn = conn.begin()
    try:
        yield conn
    finally:
        txn.rollback()
        conn.close()


def _load_all_fixtures() -> dict:
    """Read every <city>.json under fixtures/cli/ into a dict keyed by city code."""
    data: dict = {}
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        payload = json.loads(path.read_text())
        data[payload["city"]] = payload
    return data


@pytest.fixture(scope="session")
def grib_fixture():
    """Return a loader resolving a vendored GRIB2 fixture by short name.

    Mirrors the ``cli_fixture``/``FIXTURE_DIR`` idiom: ``grib_fixture("hrrr")`` →
    ``tests/fixtures/hrrr_t2m_sample.grib2`` as a ``pathlib.Path``. Tests open the
    returned path with cfgrib directly (offline, deterministic) — the GRIB decode path
    is exercised against the real eccodes binary without any network or Herbie call.
    Skips cleanly if a requested fixture is missing so a partial checkout fails loud-but-
    skippable rather than erroring at collection.
    """

    _NAMES = {
        "hrrr": "hrrr_t2m_sample.grib2",
        "gfs": "gfs_t2m_sample.grib2",
        "gefs": "gefs_p01_t2m_sample.grib2",
    }

    def _load(name: str) -> pathlib.Path:
        filename = _NAMES.get(name, name)
        path = GRIB_FIXTURE_DIR / filename
        if not path.exists():
            pytest.skip(f"GRIB fixture not found: {path}")
        return path

    return _load


@dataclass(frozen=True)
class SyntheticStratum:
    """A synthetic single-stratum draw with KNOWN EMOS/NGR params (Phase 3 calibration).

    The observations ``y`` are drawn from the TRUE predictive law under the D-02 link
    (``mu = a + b*m``, ``sigma^2 = max(sigma_floor^2, c^2 + d^2*s2)``), so a correct fitter
    must recover ``(a, b, |c|, |d|)`` (the signs of c, d are free — only c^2, d^2 enter).
    ``s2`` is the per-sample ensemble variance: a positive constant for an ensemble model,
    or all-zeros for a deterministic model (then ``d`` is inactive, D-02 — expected).
    """

    a: float
    b: float
    c: float
    d: float
    sigma_floor: float
    m: np.ndarray  # forecast ensemble mean (°F)
    s2: np.ndarray  # forecast ensemble variance
    y: np.ndarray  # verifying obs (°F), drawn from the true predictive law


def _draw_stratum(
    *,
    seed: int,
    n: int,
    a: float,
    b: float,
    c: float,
    d: float,
    sigma_floor: float,
    deterministic: bool,
) -> SyntheticStratum:
    rng = np.random.default_rng(seed)
    m = rng.normal(70.0, 8.0, n)
    s2 = np.zeros(n) if deterministic else np.full(n, 4.0)
    mu_t = a + b * m
    sig_t = np.sqrt(np.maximum(sigma_floor**2, c**2 + d**2 * s2))
    y = rng.normal(mu_t, sig_t)
    return SyntheticStratum(
        a=a, b=b, c=c, d=d, sigma_floor=sigma_floor, m=m, s2=s2, y=y
    )


@pytest.fixture
def synthetic_stratum() -> SyntheticStratum:
    """A well-sampled ENSEMBLE stratum with known ``(a, b, c, d)`` for fit-recovery tests.

    n=5000 draws under the true predictive law; ``s2 = 4.0`` constant so the spread param
    ``d`` is genuinely identifiable. Reused by 03's fit-recovery and OOS plans.
    """
    return _draw_stratum(
        seed=1, n=5000, a=1.0, b=0.95, c=1.5, d=0.8,
        sigma_floor=0.5, deterministic=False,
    )


@pytest.fixture
def synthetic_stratum_deterministic() -> SyntheticStratum:
    """A DETERMINISTIC stratum (``s2 == 0``) — ``sigma^2 = c^2`` constant, ``d`` inactive.

    Used to assert the fit still recovers ``(a, b, c)`` while ``d`` stays at init (D-02 /
    RESEARCH Pitfall 2 — expected, not a bug).
    """
    return _draw_stratum(
        seed=2, n=5000, a=2.0, b=1.02, c=2.5, d=0.0,
        sigma_floor=0.5, deterministic=True,
    )


@pytest.fixture(scope="session")
def cli_fixture() -> dict:
    """Return the vendored NWS CLI parity fixtures keyed by Kalshi city code.

    Each entry: ``{"city", "station", "std_offset_hours", "days": {"winter"|"summer":
    {"date", "cli_max", "source_url", "obs": [{"ts_utc", "temp_f"}, ...]}}}``. The obs
    are UTC-timestamped hourly temperatures; the in-window maximum equals ``cli_max``,
    and each day includes at least one just-out-of-window hotter reading so the parity
    test proves the half-open window excludes the boundary (D-03 / D-04).
    """
    fixtures = _load_all_fixtures()
    if not fixtures:
        pytest.skip("No CLI fixtures found under tests/fixtures/cli/")
    return fixtures


@dataclass(frozen=True)
class SyntheticGaussians:
    """A set of known per-model Gaussians + their analytic Vincentization blend (Phase 4).

    For Gaussians, quantile-averaging (Vincentization, D-01) has the exact closed form
    ``N(μ_blend = Σwᵢμᵢ, σ_blend = Σwᵢσᵢ)`` — ``σ_blend`` is the weighted MEAN of the σ's,
    NOT ``sqrt(Σwᵢσᵢ²)``. The fixture carries the component ``(mus, sigmas, weights)`` (with
    ``weights`` already summing to 1) and the analytic ``(mu_blend, sigma_blend)`` so the
    blend-recovery test asserts ``blend_gaussians`` reproduces them and the σ-monotonicity
    test asserts ``sigma_blend <= max(sigmas)`` (true by construction).
    """

    mus: np.ndarray  # per-model predictive means (°F)
    sigmas: np.ndarray  # per-model predictive std-devs (°F), all > 0
    weights: np.ndarray  # accuracy weights over the models present (sum to 1)
    mu_blend: float  # analytic Σwᵢμᵢ
    sigma_blend: float  # analytic Σwᵢσᵢ (weighted MEAN of σ — Vincentization, NOT RMS)


@pytest.fixture
def synthetic_gaussians() -> SyntheticGaussians:
    """Known per-model Gaussians + the analytic Vincentization blend (D-01 blend-recovery).

    Seeded via ``np.random.default_rng`` (mirroring the ``_draw_stratum`` style): draw a few
    positive σ's and means, attach random positive weights renormalized to sum to 1, and
    precompute the closed-form blend ``(Σwᵢμᵢ, Σwᵢσᵢ)``. Consumed by ``test_blend.py`` for
    blend-recovery and the σ-monotonicity invariant.
    """
    rng = np.random.default_rng(404)
    mus = rng.normal(70.0, 8.0, 4)
    sigmas = rng.uniform(1.0, 6.0, 4)
    raw_w = rng.uniform(0.1, 1.0, 4)
    weights = raw_w / raw_w.sum()
    mu_blend = float(np.dot(weights, mus))
    sigma_blend = float(np.dot(weights, sigmas))
    return SyntheticGaussians(
        mus=mus,
        sigmas=sigmas,
        weights=weights,
        mu_blend=mu_blend,
        sigma_blend=sigma_blend,
    )
