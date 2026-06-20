"""Strata layer: assemble training pairs, K→°F seam, member aggregation, and pooling.

Turns the append-only ledger into per-``(city, model, lead, month)`` fits, degrading sparse
strata smoothly toward a pooled parent rather than producing over-confident fits that would
blow up Phase-4 Kelly sizing. Model-label-generic, full-natural-key read, season-pooling
ladder, and σ-floor are all documented in docs/DECISIONS.md (D-01/D-03/D-07/D-08/D-09).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sqlalchemy.engine import RowMapping

from weatherquant.calibrate.emos import fit_stratum
from weatherquant.db import queries
from weatherquant.db.types import Bind

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

# Operational defaults (principled ASSUMPTIONS, CLI-overridable; D-12). N_MIN: parent-fallback
# floor. KAPPA: shrinkage constant in w = n/(n+KAPPA). SIGMA_FLOOR_F: °F σ-floor (D-09).
N_MIN: int = 30
KAPPA: float = 30.0
SIGMA_FLOOR_F: float = 0.5

# The ground-truth observation source (ASOS); 'afd' rows are AFD text, never a label (D-07/D-16).
OBS_SOURCE: str = "asos"

# Meteorological seasons (D-08 pooling ladder, rung 1): DJF / MAM / JJA / SON.
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
    """The ONE K→°F seam on the calibration path; params are stored in °F (D-03)."""
    return (t_k - _KELVIN_TO_CELSIUS_OFFSET) * _C_TO_F_SCALE + _C_TO_F_OFFSET


@dataclass(frozen=True)
class StratumSamples:
    """Aggregated training samples for one stratum (members already collapsed, D-02/D-07).

    ``m``/``s2`` are the per-date ensemble mean/variance (°F; ``s2=0`` deterministic), ``y`` the
    verifying daily-high obs (°F); arrays aligned one-row-per ``(city, model, lead, target_date)``.
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

    ``(a, b, c, d)`` are the EMOS link params (μ = a + b·m; σ² = max(σ_floor², c² + d²·s²));
    ``pool_level`` is the D-08 provenance: ``"month"`` / ``"shrunk:<rung>"`` / ``"parent:<rung>"``.
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

    ``m``/``s2`` ensemble mean/variance (°F); ``y`` the daily-high obs (°F); ``month`` from
    ``target_date`` (D-07). ``target_date`` is kept (not just its month) so the temporal OOS
    split (D-10) can order by real date. Member axis collapsed here.
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
        samples: the PARENT stratum's samples (coarser rung), or ``None`` when ``stratum`` is
            itself the finest rung fit on its own data. Fit here if ``parent_fit`` is absent.
        parent_fit: a pre-computed parent :class:`StratumFit` (avoids refitting per child);
            ignored when ``samples`` is ``None``.
        rung: the coarser rung this parent represents (recorded in ``pool_level``).

    Returns:
        A :class:`StratumFit`. No parent ⇒ own fit (``pool_level=rung``); ``n < N_MIN`` ⇒ parent
        verbatim (``"parent:<rung>"``); else own fit blended ``w = n/(n+KAPPA)`` (``"shrunk:<rung>"``).
    """
    if samples is None:
        return _fit_own(stratum, pool_level=rung)

    parent = parent_fit if parent_fit is not None else _fit_own(samples, pool_level=rung)

    # Below N_MIN the fine stratum is too sparse to trust — use the parent params entirely (D-08).
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
    # Blend mean params (a, b) linearly but variance params by MAGNITUDE: c, d enter σ only via
    # their squares (D-02), so signs are free and a linear blend could cancel them toward 0 —
    # collapsing σ to the floor into a spurious over-confident fit. |c|/|d| avoid that.
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

    The production pooling path: every month shrinks toward its meteorological-season parent
    (rung 1 of the D-08 ladder) so a sparse month borrows the season's spread. A season whose
    own pooled count is below ``N_MIN`` cannot anchor a trustworthy parent, so its months are
    SKIPPED (logged absence, not a degenerate fit; CR-01). Returns one
    ``(month_samples, target_dates, fit)`` per RETAINED month for the caller's OOS audit + persist.

    SHRINKAGE-TARGET NOTE (IN-02): each season parent is fit on ALL season pairs — INCLUDING the
    month shrunk toward it — so the prior is not strictly independent. Intentional this milestone:
    keeps the parent maximally data-rich; a leave-the-month-out parent is deferred (changes output,
    needs its own validation). Cost: very sparse months are mildly under-regularized.
    """
    by_month: dict[int, list[TrainingPair]] = {}
    by_season: dict[int, list[TrainingPair]] = {}
    for p in pairs:
        by_month.setdefault(p.month, []).append(p)
        by_season.setdefault(season_of(p.month), []).append(p)

    # Fit each season parent once (reused by every month). A season below N_MIN is left out so
    # its months fall through to the skip branch below.
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

    Pure (no DB): converts ``temp_kelvin`` to °F via :func:`kelvin_to_fahrenheit`, groups by
    ``(city, model, lead, target_date)`` and collapses members to ``m = mean`` / ``s2 = population
    variance`` in Python (member collapse happens HERE, after the full-key read — D-02), joins to
    the verifying obs on ``(city, target_date)`` (``y = daily_high_f``), and derives ``month`` (D-07).

    Rows with no matching obs, null ``daily_high_f``, or null ``temp_kelvin`` are skipped (never
    interpolated). Model-label-generic: no per-model branching.
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
                target_date=target_date,  # real verifying day — for the temporal split (D-10)
                m=float(arr.mean()),
                s2=float(arr.var()),  # population variance over members; 0 for one member
                y=y,
            )
        )
    return pairs


def assemble_training_pairs(
    bind: Bind,
    *,
    city: str,
    model: str,
) -> list[TrainingPair]:
    """Read forecasts + observations from the ledger and assemble training pairs (D-01/D-03).

    Reads ``forecasts`` via :func:`weatherquant.db.queries.latest` with the FULL canonical key
    (NEVER a strict subset, which ``latest()`` rejects — the ensemble-collapse trap), scoped to
    one ``(city, model)`` via the injection-safe ``where=`` filter (``model`` never f-stringed
    into SQL — T-03-05). Observations read for the same ``city`` with ``source='asos'``.
    Aggregation + K→°F seam + join happen in :func:`assemble_pairs_from_rows`. Model-label-generic
    (D-01): ``model`` passes straight through as a value, never branched per label.
    """
    forecast_rows = queries.latest(
        bind, "forecasts", where={"city": city, "model": model}
    )
    obs_rows = queries.latest(
        bind, "observations", where={"city": city, "source": OBS_SOURCE}
    )
    return assemble_pairs_from_rows(forecast_rows, obs_rows)
