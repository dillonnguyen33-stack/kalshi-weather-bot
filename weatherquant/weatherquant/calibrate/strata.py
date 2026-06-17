"""The strata layer: assemble training pairs + K→°F seam + member aggregation + pooling.

This module turns the append-only ledger into per-``(city, model, lead, month)`` calibrated
fits. It is the boundary where Kelvin forecasts become °F, ensemble members collapse into a
``(mean, variance)`` predictor, and sparse strata degrade *smoothly* toward a pooled parent
instead of producing degenerate over-confident fits (which would blow up Phase-4 Kelly sizing).

**Model-label-generic (D-01).** The read → aggregate → fit path keys on whatever ``model``
string is present — the four NOAA models (``nbm``/``hrrr``/``gfs``/``gefs``) AND the
supplementary blend inputs (``nws``/``wethr:*``/``openmeteo[:member]``). There is NO per-model
``if model ==`` branching for the calibration math: a supplementary label calibrates through
the exact same code as ``gfs``.

**The K→°F seam (D-03).** :func:`kelvin_to_fahrenheit` is THE single named K→°F conversion on
the calibration path — mirroring ``ingest/obs.py::celsius_to_fahrenheit``. No other module in
``calibrate/`` inlines the ``9/5`` / ``459.67`` arithmetic, so the units boundary stays
auditable in one place.

**Full natural key (WR-02 trap, RESEARCH Pitfall 3).** Forecasts are read with the FULL
canonical key (``city, target_date, model, lead, member``) — the ``db.queries.latest`` default.
An under-specified (strict-subset) key would DISTINCT-ON a narrower tuple and silently collapse
distinct ensemble members into one wrong "current truth"; ``latest()`` rejects that with
``ValueError``. Members are aggregated to ``(m = mean, s2 = variance)`` in Python *after* the
read, per ``(city, model, lead, target_date)``; ``month`` is derived from ``target_date`` (D-07).

**Pooling ladder + shrinkage (D-08).** When a fine stratum is data-starved we coarsen the most
data-starved axis first — ``(city,model,lead,month)`` → ``(city,model,lead, season)`` →
``(city,model, lead pooled to adjacent leads)`` (RESEARCH Open Question #3: adjacent lead values
within the same ``(city,model)``). For a fine stratum with ``n`` samples and a parent fit:

* ``n < N_MIN``  → use the parent params *entirely* (record ``"parent:<rung>"``).
* ``n >= N_MIN`` → fit the fine stratum and blend toward the parent with own-weight
  ``w = n / (n + KAPPA)`` (record ``"shrunk:<rung>"``).
* no parent (the finest rung itself) → the own fit (record the rung name, e.g. ``"month"``).

The pooling **level used is recorded** per stratum (``pool_level``) for audit.

**σ-floor (D-09).** Predictive σ is clamped to ``>= SIGMA_FLOOR_F`` and the variance-param
gradient is masked when the floor is active — both enforced inside the link.py
``predict`` / ``param_grads`` reused by :func:`weatherquant.calibrate.emos.fit_stratum`.

The hyperparameters ``N_MIN``, ``KAPPA``, ``SIGMA_FLOOR_F`` are principled, research-ranged
defaults (RESEARCH §"Operational Defaults") — ASSUMPTIONS for this dataset, CLI-overridable, and
tuned (if ever) ONLY on a train/val split disjoint from the Phase-6 Gate-1 slice (anti-p-hacking,
D-12). Pure NumPy + stdlib — no scipy/sklearn (the AST guard enforces it).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sqlalchemy.engine import Connection, Engine, RowMapping

from weatherquant.calibrate.emos import fit_stratum
from weatherquant.db import queries

__all__ = [
    "kelvin_to_fahrenheit",
    "season_of",
    "N_MIN",
    "KAPPA",
    "SIGMA_FLOOR_F",
    "StratumSamples",
    "StratumFit",
    "fit_stratum_pooled",
    "fit_pooled_month_strata",
    "assemble_pairs_from_rows",
    "assemble_training_pairs",
    "OBS_SOURCE",
]

logger = logging.getLogger(__name__)

# --- Operational defaults (RESEARCH §"Operational Defaults"; principled ASSUMPTIONS, D-12) -
# N_MIN: hard floor below which a stratum falls back to the parent entirely.
# KAPPA: shrinkage constant in w = n/(n+KAPPA) — own-weight is ~50% at n≈KAPPA.
# SIGMA_FLOOR_F: the °F σ-floor blocking degenerate over-confidence (D-09).
N_MIN: int = 30
KAPPA: float = 30.0
SIGMA_FLOOR_F: float = 0.5

# The ground-truth observation source (ASOS); 'afd' rows are AFD text, never a label (D-07/D-16).
OBS_SOURCE: str = "asos"

# Meteorological seasons (D-08 pooling ladder, rung 1): DJF / MAM / JJA / SON. A data-starved
# month stratum coarsens to its season parent before any finer-grained own-fit is trusted.
_SEASON_BY_MONTH: dict[int, int] = {
    12: 0, 1: 0, 2: 0,  # DJF
    3: 1, 4: 1, 5: 1,  # MAM
    6: 2, 7: 2, 8: 2,  # JJA
    9: 3, 10: 3, 11: 3,  # SON
}


def season_of(month: int) -> int:
    """Meteorological season index (0=DJF, 1=MAM, 2=JJA, 3=SON) for the pooling ladder (D-08)."""
    return _SEASON_BY_MONTH[month]

# Absolute-zero offset and the °F scale — used ONLY inside the single K→°F seam below.
_KELVIN_TO_CELSIUS_OFFSET = 273.15
_C_TO_F_SCALE = 9.0 / 5.0
_C_TO_F_OFFSET = 32.0


def kelvin_to_fahrenheit(t_k: float) -> float:
    """The ONE K→°F seam on the calibration path (D-03).

    Mirrors ``ingest/obs.py::celsius_to_fahrenheit``: keeping the conversion in a single named
    function means no other module under ``calibrate/`` inlines ``273.15`` / ``9/5`` / ``459.67``
    and the units boundary (Kelvin forecasts → °F calibration space) stays auditable in one
    place. ``calibration_params`` are stored in °F space (D-03).
    """
    return (t_k - _KELVIN_TO_CELSIUS_OFFSET) * _C_TO_F_SCALE + _C_TO_F_OFFSET


@dataclass(frozen=True)
class StratumSamples:
    """Aggregated training samples for one stratum (members already collapsed, D-02/D-07).

    ``m`` is the per-date ensemble-mean forecast (°F), ``s2`` the per-date ensemble variance
    (``0`` for a deterministic model), ``y`` the verifying daily-high obs (°F). The arrays are
    aligned and one-row-per ``(city, model, lead, target_date)`` — ``month`` is the calendar
    month of those target dates (a single value per stratum, D-07).
    """

    city: str
    model: str
    lead: int
    month: int
    m: NDArray[np.float64]
    s2: NDArray[np.float64]
    y: NDArray[np.float64]

    @property
    def n(self) -> int:
        """Number of training samples in this stratum."""
        return int(len(self.y))


@dataclass(frozen=True)
class StratumFit:
    """A fitted stratum's params + provenance, ready for the calibration_params payload (D-13).

    ``(a, b, c, d)`` are the EMOS link params (μ = a + b·m; σ² = max(σ_floor², c² + d²·s²)),
    ``sigma_floor`` the °F clamp (D-09), ``n_train`` the samples used, and ``pool_level`` the
    pooling-ladder provenance string (D-08): ``"month"`` (finest, no pooling), ``"shrunk:<rung>"``
    (blended toward the parent), or ``"parent:<rung>"`` (n<N_MIN, parent used entirely).
    """

    city: str
    model: str
    lead: int
    month: int
    a: float
    b: float
    c: float
    d: float
    sigma_floor: float
    n_train: int
    pool_level: str


@dataclass(frozen=True)
class TrainingPair:
    """One aggregated (forecast, verifying-obs) training row for a target date.

    ``m``/``s2`` are the ensemble mean/variance over members (°F); ``y`` the daily-high obs (°F);
    ``month`` is derived from ``target_date`` (D-07). ``target_date`` is the verifying day itself —
    kept (not just its month) so a temporal OOS split (D-10) can order samples by real date rather
    than collapsing them to a single synthetic key. The member axis is gone — collapsed here.
    """

    city: str
    model: str
    lead: int
    month: int
    target_date: date
    m: float
    s2: float
    y: float


def _fit_own(stratum: StratumSamples, pool_level: str) -> StratumFit:
    """Fit a stratum on its own samples (no pooling) — the finest-rung / parent fit."""
    a, b, c, d = fit_stratum(stratum.m, stratum.s2, stratum.y, sigma_floor=SIGMA_FLOOR_F)
    return StratumFit(
        city=stratum.city,
        model=stratum.model,
        lead=stratum.lead,
        month=stratum.month,
        a=a,
        b=b,
        c=c,
        d=d,
        sigma_floor=SIGMA_FLOOR_F,
        n_train=stratum.n,
        pool_level=pool_level,
    )


def fit_stratum_pooled(
    stratum: StratumSamples,
    *,
    samples: StratumSamples | None = None,
    parent_fit: StratumFit | None = None,
    rung: str = "month",
) -> StratumFit:
    """Fit one stratum, pooling/shrinking toward a parent when sparse (CAL-03 / D-08).

    Args:
        stratum: the fine stratum's aggregated samples.
        samples: the PARENT stratum's aggregated samples (the coarser rung), or ``None`` when
            ``stratum`` is itself the finest rung being fit on its own data. When given without
            ``parent_fit`` the parent is fit here.
        parent_fit: a pre-computed parent :class:`StratumFit` (avoids refitting the parent for
            every child); ignored when ``samples`` is ``None``.
        rung: the coarser rung this parent represents (``"month"`` default; ``"season"`` or
            ``"lead-neighbors"`` higher up the ladder) — recorded in ``pool_level``.

    Returns:
        A :class:`StratumFit`. With no parent: the own fit, ``pool_level=rung``. With a parent:
        ``n < N_MIN`` ⇒ parent params verbatim (``pool_level="parent:<rung>"``); otherwise the
        own fit blended toward the parent with ``w = n/(n+KAPPA)`` (``pool_level="shrunk:<rung>"``).
    """
    # Finest rung — no parent to pool toward.
    if samples is None:
        return _fit_own(stratum, pool_level=rung)

    parent = parent_fit if parent_fit is not None else _fit_own(samples, pool_level=rung)

    # Hard fallback: below N_MIN the fine stratum is too sparse to trust at all — use the
    # parent params ENTIRELY (D-08), recording the parent provenance.
    if stratum.n < N_MIN:
        return StratumFit(
            city=stratum.city,
            model=stratum.model,
            lead=stratum.lead,
            month=stratum.month,
            a=parent.a,
            b=parent.b,
            c=parent.c,
            d=parent.d,
            sigma_floor=SIGMA_FLOOR_F,
            n_train=stratum.n,
            pool_level=f"parent:{rung}",
        )

    # Enough data to fit, but still shrink toward the parent: own-weight w = n/(n+KAPPA).
    own = _fit_own(stratum, pool_level=rung)
    w = stratum.n / (stratum.n + KAPPA)
    # Mean params (a, b) blend linearly. Variance params (c, d) enter the predictive σ ONLY
    # through their squares (σ² = c² + d²·s², link.predict / D-02), so their signs are free:
    # the fitter may return c<0 for the child and c>0 for the parent for the SAME spread. A
    # linear blend would then cancel them toward 0, collapsing σ to the floor — a spuriously
    # over-confident fit, the exact Kelly-blowup the pooling ladder exists to prevent. Blend
    # the MAGNITUDES instead (|c|, |d| are equivalent representations of the same variance).
    return StratumFit(
        city=stratum.city,
        model=stratum.model,
        lead=stratum.lead,
        month=stratum.month,
        a=w * own.a + (1.0 - w) * parent.a,
        b=w * own.b + (1.0 - w) * parent.b,
        c=w * abs(own.c) + (1.0 - w) * abs(parent.c),
        d=w * abs(own.d) + (1.0 - w) * abs(parent.d),
        sigma_floor=SIGMA_FLOOR_F,
        n_train=stratum.n,
        pool_level=f"shrunk:{rung}",
    )


def _samples_from_pairs(
    pairs: Sequence[TrainingPair], *, city: str, model: str, lead: int, month: int
) -> StratumSamples:
    """Collapse aligned :class:`TrainingPair` rows into one :class:`StratumSamples` (m, s2, y)."""
    return StratumSamples(
        city=city,
        model=model,
        lead=lead,
        month=month,
        m=np.array([p.m for p in pairs], dtype=float),
        s2=np.array([p.s2 for p in pairs], dtype=float),
        y=np.array([p.y for p in pairs], dtype=float),
    )


def fit_pooled_month_strata(
    pairs: Sequence[TrainingPair], *, city: str, model: str, lead: int
) -> list[tuple[StratumSamples, list[date], StratumFit]]:
    """Fit every month stratum for one ``(city, model, lead)``, pooling toward season (CR-01/D-08).

    The production pooling path. The CLI previously called :func:`fit_stratum_pooled` with no
    parent, so the ``n < N_MIN`` parent-fallback and the shrinkage blend never fired outside
    tests — a data-starved month was persisted as its own degenerate over-confident fit. Here
    every month stratum shrinks toward its meteorological-season parent (rung 1 of the D-08
    ladder), so a sparse month borrows the season's spread instead of inventing a false-sharp one.

    A season whose own pooled sample count is below ``N_MIN`` cannot anchor a trustworthy parent,
    so its months are SKIPPED (a logged absence, not a degenerate fit) — the explicit fail-safe
    the review's CR-01 step 5 calls for. Phase-6 lead-neighbor coarsening (rung 2) is not wired
    yet; until it is, those sparse-season city-months simply have no calibration row.

    Returns one ``(month_samples, target_dates, fit)`` per RETAINED month so the caller can run
    the OOS audit and persist using the month's own samples.
    """
    by_month: dict[int, list[TrainingPair]] = {}
    by_season: dict[int, list[TrainingPair]] = {}
    for p in pairs:
        by_month.setdefault(p.month, []).append(p)
        by_season.setdefault(season_of(p.month), []).append(p)

    # Fit each season parent once (reused by every month in the season). A season below N_MIN is
    # left out of the map so its months fall through to the skip branch below.
    season_parents: dict[int, tuple[StratumSamples, StratumFit]] = {}
    for season, season_pairs in by_season.items():
        if len(season_pairs) < N_MIN:
            continue
        parent_samples = _samples_from_pairs(
            season_pairs, city=city, model=model, lead=lead, month=season_pairs[0].month
        )
        season_parents[season] = (parent_samples, _fit_own(parent_samples, pool_level="season"))

    out: list[tuple[StratumSamples, list[date], StratumFit]] = []
    for month, month_pairs in sorted(by_month.items()):
        parent = season_parents.get(season_of(month))
        if parent is None:
            logger.warning(
                "calibrate skip city=%s model=%s lead=%d month=%d: season parent n<%d — "
                "too sparse for a trustworthy fit (D-08), persisting nothing",
                city,
                model,
                lead,
                month,
                N_MIN,
            )
            continue
        parent_samples, parent_fit = parent
        month_samples = _samples_from_pairs(
            month_pairs, city=city, model=model, lead=lead, month=month
        )
        fit = fit_stratum_pooled(
            month_samples, samples=parent_samples, parent_fit=parent_fit, rung="season"
        )
        target_dates = [p.target_date for p in month_pairs]
        out.append((month_samples, target_dates, fit))
    return out


def assemble_pairs_from_rows(
    forecast_rows: Sequence[RowMapping | Mapping[str, Any]],
    obs_rows: Sequence[RowMapping | Mapping[str, Any]],
) -> list[TrainingPair]:
    """Join forecasts↔observations and aggregate members to ``(mean, variance)`` (D-02/D-03).

    Pure (no DB): takes the rows already read via :func:`weatherquant.db.queries.latest` and

    1. converts each forecast ``temp_kelvin`` to °F through the single :func:`kelvin_to_fahrenheit`
       seam,
    2. groups forecasts by ``(city, model, lead, target_date)`` and aggregates the ensemble
       MEMBERS to ``m = mean`` and ``s2 = population variance`` (``s2 == 0`` for a single-member
       deterministic model) — the member collapse happens HERE, in Python, after the full-key
       read (RESEARCH Pitfall 3),
    3. joins to the verifying observation on ``(city, target_date)`` (``y = daily_high_f``, °F),
    4. derives ``month`` from ``target_date`` (D-07).

    Rows with no matching observation, a null ``daily_high_f``, or a null ``temp_kelvin`` are
    skipped (absence is absence — never interpolated). Model-label-generic: no per-model branching.
    """
    # Label lookup: (city, target_date) -> daily_high_f (°F), ASOS only.
    labels: dict[tuple[Any, Any], float] = {}
    for row in obs_rows:
        if row.get("source") not in (None, OBS_SOURCE):
            continue
        y = row.get("daily_high_f")
        if y is None:
            continue
        labels[(row["city"], row["target_date"])] = float(y)

    # Group forecast members by (city, model, lead, target_date).
    groups: dict[tuple[Any, Any, Any, Any], list[float]] = {}
    for row in forecast_rows:
        t_k = row.get("temp_kelvin")
        if t_k is None:
            continue
        key = (row["city"], row["model"], row["lead"], row["target_date"])
        groups.setdefault(key, []).append(kelvin_to_fahrenheit(float(t_k)))

    pairs: list[TrainingPair] = []
    for (city, model, lead, target_date), members_f in groups.items():
        y = labels.get((city, target_date))
        if y is None:
            continue  # no verifying obs — drop the pair (no look-ahead, no interpolation)
        arr = np.asarray(members_f, dtype=float)
        pairs.append(
            TrainingPair(
                city=city,
                model=model,
                lead=int(lead),
                month=int(target_date.month),  # D-07
                target_date=target_date,  # the real verifying day — for the temporal split (D-10)
                m=float(arr.mean()),
                s2=float(arr.var()),  # population variance over members; 0 for one member
                y=y,
            )
        )
    return pairs


def assemble_training_pairs(
    bind: Engine | Connection,
    *,
    city: str,
    model: str,
) -> list[TrainingPair]:
    """Read forecasts + observations from the ledger and assemble training pairs (D-01/D-03).

    Reads ``forecasts`` via :func:`weatherquant.db.queries.latest` with the FULL canonical key
    (the default — NEVER a strict subset, which ``latest()`` rejects: the WR-02 ensemble-collapse
    trap, RESEARCH Pitfall 3), scoped to one ``(city, model)`` via the injection-safe ``where=``
    equality filter (column names resolve through ``table.c[name]``; ``model`` is never f-stringed
    into SQL — threat T-03-05). Observations are read for the same ``city`` with ``source='asos'``
    (excluding ``source='afd'``). Member aggregation + the K→°F seam + the join happen in
    :func:`assemble_pairs_from_rows`.

    Model-label-generic (D-01): ``model`` is passed straight through as a value — the four NOAA
    models and the supplementary inputs (``nws``/``wethr:*``/``openmeteo``) all calibrate through
    this one path, never branched per label.
    """
    forecast_rows = queries.latest(
        bind, "forecasts", where={"city": city, "model": model}
    )
    obs_rows = queries.latest(
        bind, "observations", where={"city": city, "source": OBS_SOURCE}
    )
    return assemble_pairs_from_rows(forecast_rows, obs_rows)
