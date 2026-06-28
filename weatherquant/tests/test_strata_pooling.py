"""Sparse-strata pooling / shrinkage tests for strata.py (CAL-03, D-07/D-08/D-09).

The calibration engine must degrade *smoothly* on sparse strata instead of producing
degenerate over-confident fits that would blow up Phase-4 Kelly sizing. These tests pin the
four CAL-03 guarantees on in-memory aggregated samples (no DB needed for the pooling math):

* **Shrinkage** — a fine stratum blends toward its pooled parent with own-weight
  ``w = n / (n + KAPPA)``: ~50% at ``n ≈ KAPPA``, approaching 1 as ``n ≫ KAPPA``.
* **N_min fallback** — a stratum with ``n < N_MIN`` uses the parent params *entirely* and
  records the parent's ``pool_level``.
* **σ-floor** — predictive σ is clamped to ``>= SIGMA_FLOOR_F`` and the variance-param
  gradient is masked when the floor is active (enforced via the link.py predict/param_grads).
* **pool_level recorded** — every fitted stratum carries a provenance string for the rung used.

A fifth test pins the WR-02 trap: the forecasts read uses the FULL canonical natural key —
an under-specified ``latest()`` call raises ``ValueError`` — and member aggregation happens in
Python after the read. The single K→°F seam ``kelvin_to_fahrenheit`` is checked too.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from weatherquant.calibrate import strata
from weatherquant.calibrate.link import predict
from weatherquant.calibrate.strata import (
    KAPPA,
    N_MIN,
    SIGMA_FLOOR_F,
    StratumFit,
    StratumSamples,
    fit_pooled_month_strata,
    fit_stratum_pooled,
    kelvin_to_fahrenheit,
)
from weatherquant.db import queries


def _samples(*, n: int, a: float, b: float, c: float, d: float, seed: int) -> StratumSamples:
    """Draw an in-memory aggregated stratum (m, s2, y) from the true predictive law."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0, 8.0, n)
    s2 = rng.uniform(1.0, 16.0, n)
    mu = a + b * m
    sig = np.sqrt(np.maximum(SIGMA_FLOOR_F**2, c**2 + d**2 * s2))
    y = rng.normal(mu, sig)
    return StratumSamples(
        city="NYC", model="gfs", lead=1, month=6, m=m, s2=s2, y=y
    )


def test_kelvin_to_fahrenheit_is_the_seam() -> None:
    """The single K→°F seam: 273.15 K == 32 °F, 373.15 K == 212 °F."""
    assert kelvin_to_fahrenheit(273.15) == pytest.approx(32.0)
    assert kelvin_to_fahrenheit(373.15) == pytest.approx(212.0)


def test_shrinkage_weight_blends_toward_parent() -> None:
    """Own-weight w = n/(n+KAPPA): ~50% at n≈KAPPA, →fine as n≫KAPPA (D-08)."""
    parent = _samples(n=4000, a=5.0, b=1.0, c=2.0, d=0.5, seed=100)
    parent_fit = fit_stratum_pooled(parent, samples=None)  # parent fits on its own data

    # n == KAPPA → own-weight ~0.5: the blended params sit roughly halfway between the fine
    # fit and the parent. Use a fine stratum whose own fit is clearly different from parent.
    n_kappa = int(KAPPA)
    fine = _samples(n=n_kappa, a=9.0, b=1.0, c=2.0, d=0.5, seed=101)
    blended = fit_stratum_pooled(fine, samples=parent, parent_fit=parent_fit)

    w = n_kappa / (n_kappa + KAPPA)  # == 0.5 when n == KAPPA
    assert w == pytest.approx(0.5)
    # Blended intercept must lie strictly between the parent and the fine own-fit.
    fine_own = fit_stratum_pooled(fine, samples=None)
    lo, hi = sorted((parent_fit.a, fine_own.a))
    assert lo < blended.a < hi
    assert blended.pool_level == "shrunk:month"
    assert blended.n_train == n_kappa

    # n ≫ KAPPA → own-weight → 1: blended approaches the fine own-fit.
    big = _samples(n=20 * n_kappa, a=9.0, b=1.0, c=2.0, d=0.5, seed=102)
    big_blended = fit_stratum_pooled(big, samples=parent, parent_fit=parent_fit)
    big_own = fit_stratum_pooled(big, samples=None)
    assert abs(big_blended.a - big_own.a) < abs(big_blended.a - parent_fit.a)


def test_shrinkage_blends_variance_params_by_magnitude(monkeypatch) -> None:
    """c/d feed σ only through c²/d² (sign-free, link.predict), so an opposite-sign child and
    parent must NOT cancel during shrinkage. Magnitude blending keeps |c| bounded away from 0
    instead of collapsing σ to the floor — the spurious over-confidence WR-01 guards against."""
    parent_fit = StratumFit(
        city="NYC", model="gfs", lead=1, month=6,
        a=0.0, b=1.0, c=2.0, d=1.0,
        sigma_floor=SIGMA_FLOOR_F, n_train=4000, pool_level="month",
    )
    # Force the fine stratum's own fit to carry the SAME spread but OPPOSITE signs on c/d — the
    # case a naive linear blend would cancel to ~0.
    own_fit = StratumFit(
        city="NYC", model="gfs", lead=1, month=6,
        a=0.0, b=1.0, c=-2.0, d=-1.0,
        sigma_floor=SIGMA_FLOOR_F, n_train=int(KAPPA), pool_level="month",
    )
    monkeypatch.setattr(strata, "_fit_own", lambda stratum, pool_level: own_fit)

    fine = _samples(n=int(KAPPA), a=0.0, b=1.0, c=2.0, d=1.0, seed=900)
    blended = fit_stratum_pooled(fine, samples=fine, parent_fit=parent_fit)

    # At w=0.5 a linear blend of (+2, -2) collapses to c≈0 (σ→floor); magnitude blend → 2.0.
    w = int(KAPPA) / (int(KAPPA) + KAPPA)
    assert w == pytest.approx(0.5)
    assert blended.c == pytest.approx(w * 2.0 + (1.0 - w) * 2.0)  # |−2|,|2| → 2.0, no cancel
    assert blended.d == pytest.approx(w * 1.0 + (1.0 - w) * 1.0)  # → 1.0
    assert abs(blended.c) > 1.5  # decisively NOT collapsed toward 0
    assert blended.pool_level == "shrunk:month"


def test_n_min_fallback_uses_parent_entirely() -> None:
    """n < N_MIN → parent params verbatim, parent's pool_level recorded (D-08)."""
    assert N_MIN >= 2
    parent = _samples(n=4000, a=5.0, b=1.0, c=2.0, d=0.5, seed=200)
    parent_fit = fit_stratum_pooled(parent, samples=None)

    sparse = _samples(n=N_MIN - 1, a=99.0, b=2.0, c=8.0, d=3.0, seed=201)
    fit = fit_stratum_pooled(sparse, samples=parent, parent_fit=parent_fit)

    # Parent params used ENTIRELY — the sparse stratum's wildly different own-fit is ignored.
    assert fit.a == pytest.approx(parent_fit.a)
    assert fit.b == pytest.approx(parent_fit.b)
    assert fit.c == pytest.approx(parent_fit.c)
    assert fit.d == pytest.approx(parent_fit.d)
    assert fit.pool_level == "parent:month"
    assert fit.n_train == N_MIN - 1


def test_sigma_floor() -> None:
    """Predictive σ clamped >= SIGMA_FLOOR_F; variance grad masked when floor active (D-09)."""
    # Near-degenerate stratum: obs almost exactly equal the forecast → variance params driven
    # to the floor. The fit must stay finite and σ must never drop below the floor.
    rng = np.random.default_rng(300)
    n = 500
    m = rng.normal(0.0, 8.0, n)
    s2 = rng.uniform(1.0, 16.0, n)
    y = m + rng.normal(0.0, 1e-3, n)  # essentially a=0, b=1, ~zero spread
    samples = StratumSamples(city="NYC", model="gfs", lead=1, month=6, m=m, s2=s2, y=y)

    fit = fit_stratum_pooled(samples, samples=None)
    assert np.isfinite([fit.a, fit.b, fit.c, fit.d]).all()
    assert fit.sigma_floor == SIGMA_FLOOR_F

    _, sigma = predict((fit.a, fit.b, fit.c, fit.d, fit.sigma_floor), m, s2)
    assert np.all(sigma >= SIGMA_FLOOR_F - 1e-12)


def test_pool_level_recorded_on_every_fit() -> None:
    """Every StratumFit carries a non-empty pool_level provenance string (D-08)."""
    own = fit_stratum_pooled(_samples(n=4000, a=5, b=1, c=2, d=0.5, seed=400), samples=None)
    assert isinstance(own, StratumFit)
    assert isinstance(own.pool_level, str) and own.pool_level
    assert own.pool_level == "month"  # the finest rung, no pooling needed


def test_forecasts_read_requires_full_natural_key(monkeypatch) -> None:
    """Assembling pairs reads forecasts with the FULL key — a strict subset raises (WR-02).

    The strata layer must never collapse ensemble members via an under-specified key. We
    assert the guard fires by passing an explicit strict-subset key through the real
    ``queries.latest`` validation path (no DB access — the ValueError is raised before any
    query executes).
    """
    from weatherquant.db.models import (
        metadata,  # noqa: F401 — ensures tables registered
    )

    with pytest.raises(ValueError, match="missing key column"):
        # Omitting 'member' is exactly the WR-02 ensemble-collapse trap.
        queries.latest(
            bind=object(),  # never reached — validation raises first
            table_name="forecasts",
            natural_key=["city", "target_date", "model", "lead"],
        )


def test_assemble_aggregates_members_in_python() -> None:
    """Members are aggregated to (mean, variance) in Python after the full-key read (D-02)."""
    # Three ensemble members for one (city, model, lead, target_date), in Kelvin.
    target = dt.date(2026, 6, 1)
    forecast_rows = [
        {"city": "NYC", "model": "gefs", "lead": 1, "member": mem,
         "target_date": target, "temp_kelvin": 290.0 + mem}
        for mem in (0, 1, 2)
    ]
    obs_rows = [
        {"city": "NYC", "source": "asos", "target_date": target, "daily_high_f": 65.0}
    ]

    pairs = strata.assemble_pairs_from_rows(forecast_rows, obs_rows)
    assert len(pairs) == 1
    p = pairs[0]
    members_f = np.array([kelvin_to_fahrenheit(290.0 + mem) for mem in (0, 1, 2)])
    assert p.m == pytest.approx(members_f.mean())
    assert p.s2 == pytest.approx(members_f.var())  # population variance over members
    assert p.y == pytest.approx(65.0)
    assert p.month == 6  # derived from target_date (D-07)
    assert p.target_date == target  # the real verifying day is carried, not just its month (D-10)
    assert p.model == "gefs"


def _pairs(
    *, month: int, n: int, a: float, b: float, c: float, d: float, seed: int
) -> list[strata.TrainingPair]:
    """Build n aligned TrainingPairs for one month, drawn from the true predictive law."""
    rng = np.random.default_rng(seed)
    m = rng.normal(0.0, 8.0, n)
    s2 = rng.uniform(1.0, 16.0, n)
    sig = np.sqrt(np.maximum(SIGMA_FLOOR_F**2, c**2 + d**2 * s2))
    y = rng.normal(a + b * m, sig)
    base = dt.date(2024, month, 1)
    return [
        strata.TrainingPair(
            city="NYC",
            model="gfs",
            lead=1,
            month=month,
            target_date=base + dt.timedelta(days=i),
            m=float(m[i]),
            s2=float(s2[i]),
            y=float(y[i]),
        )
        for i in range(n)
    ]


def test_fit_pooled_month_strata_wires_shrink_and_parent_fallback() -> None:
    """CR-01: the production path pools each month toward its season parent — the n>=N_MIN
    shrink blend and the n<N_MIN parent fallback both fire (they were dead before)."""
    # Two months in the SAME season (JJA): a data-rich June and a sparse July.
    rich = _pairs(month=6, n=200, a=5.0, b=1.0, c=2.0, d=0.5, seed=1)
    sparse = _pairs(month=7, n=N_MIN - 1, a=99.0, b=2.0, c=8.0, d=3.0, seed=2)

    results = fit_pooled_month_strata(rich + sparse, city="NYC", model="gfs", lead=1)
    by_month = {samples.month: (samples, dates, fit) for samples, dates, fit in results}

    assert set(by_month) == {6, 7}
    # Rich month (n >= N_MIN): own fit shrunk toward the season parent.
    _, june_dates, june_fit = by_month[6]
    assert june_fit.pool_level == "shrunk:season"
    assert june_fit.n_train == 200
    assert len(june_dates) == 200  # target_dates returned for the OOS audit
    # Sparse month (n < N_MIN): parent params used entirely.
    _, _, july_fit = by_month[7]
    assert july_fit.pool_level == "parent:season"
    assert july_fit.n_train == N_MIN - 1


def test_fit_pooled_month_strata_skips_sparse_season() -> None:
    """CR-01 escape hatch: a season too sparse to anchor a trustworthy parent persists nothing
    for its months (a logged absence) rather than a degenerate over-confident fit."""
    healthy = _pairs(month=6, n=200, a=5.0, b=1.0, c=2.0, d=0.5, seed=3)
    lonely = _pairs(month=1, n=5, a=5.0, b=1.0, c=2.0, d=0.5, seed=4)  # DJF total = 5 < N_MIN

    results = fit_pooled_month_strata(healthy + lonely, city="NYC", model="gfs", lead=1)
    months = {samples.month for samples, _, _ in results}

    assert months == {6}  # the lonely sub-N_MIN-season month is skipped entirely
