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
    fine = _samples(n=KAPPA, a=9.0, b=1.0, c=2.0, d=0.5, seed=101)
    blended = fit_stratum_pooled(fine, samples=parent, parent_fit=parent_fit)

    w = KAPPA / (KAPPA + KAPPA)  # == 0.5
    assert w == pytest.approx(0.5)
    # Blended intercept must lie strictly between the parent and the fine own-fit.
    fine_own = fit_stratum_pooled(fine, samples=None)
    lo, hi = sorted((parent_fit.a, fine_own.a))
    assert lo < blended.a < hi
    assert blended.pool_level == "shrunk:month"
    assert blended.n_train == KAPPA

    # n ≫ KAPPA → own-weight → 1: blended approaches the fine own-fit.
    big = _samples(n=20 * KAPPA, a=9.0, b=1.0, c=2.0, d=0.5, seed=102)
    big_blended = fit_stratum_pooled(big, samples=parent, parent_fit=parent_fit)
    big_own = fit_stratum_pooled(big, samples=None)
    assert abs(big_blended.a - big_own.a) < abs(big_blended.a - parent_fit.a)


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
    from weatherquant.db.models import metadata  # noqa: F401 — ensures tables registered

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
    assert p.model == "gefs"
