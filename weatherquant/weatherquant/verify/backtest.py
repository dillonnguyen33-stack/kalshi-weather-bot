"""Walk-forward paired backtest (VER-03 / D-08 / D-09 / D-10): as-of-correct WQ-vs-v3 records.

D-08 (verify subtree-local — walk-forward as-of orchestration, NO new math): ``walk_forward``
replays the ledger day-by-day, building one ``PairedRecord`` per (day, city, bucket) that scores
the Weatherquant blended probability against the legacy v3 probability on the SAME realized
outcome. It is PURE ORCHESTRATION: it composes the existing Phase-3/4 primitives verbatim
(``calibrate.strata.fit_pooled_month_strata`` + ``calibrate.link.predict`` for the EMOS refit,
``price.blend.blend_gaussians`` + ``price.buckets.bucket_probs`` for the WQ arm, and
``verify.v3_reference.v3_bucket_probs`` for the v3 arm) and invents NO new temporal or
calibration logic (D-04, RESEARCH §Pattern 5). It re-derives no calibration / blend / CDF math.

NO-LOOK-AHEAD (T-06-12, RESEARCH §Pitfall 1 — the proof-killer): for each LST settlement day D
the decision cutoff is the pre-registered primary lead's fixed cutoff BEFORE D (D-05). The as-of
read consumes ONLY ledger rows with ``available_at < cutoff`` — the cutoff filter is applied
INSIDE the per-day loop (filtering after a bare ``latest()`` leaks). ``observations.daily_high_f``
is read ONLY to score the binary bucket outcome ``o_i`` (via the ``price.buckets._HALF`` edge
rule), NEVER as a training feature.

D-10 (window disjointness, anti-p-hacking): :func:`assert_window_disjoint` fails loud if the
Gate-1 ``[start, end)`` test window overlaps the Phase-3 OOS slice (D-12 in ``evaluate.py``);
it is called at the TOP of :func:`walk_forward`, before any scoring or ledger access.

D-09 (coverage logging — absence is absence): voided / missing / non-settling days are NOT
silently dropped (no bare ``dropna()`` on the day axis) — each is appended to the returned
coverage log as ``{day, city, reason}`` AND surfaced via ``logger.warning`` so excluded-day
coverage is auditable in the final verdict (VER-06).

POINT-IN-TIME v3 BIAS (D-02/D-03 integrity contract, RESEARCH Pattern 1 / Assumption A4): the v3
arm's ``corrected_mean``/``spread`` are derived from the SAME ledger ensemble rows the WQ arm sees
(per-date ``m`` / ``sqrt(s2)`` via ``strata.assemble_pairs_from_rows``) plus a POINT-IN-TIME bias
measured ONLY from ``available_at < cutoff`` settled outcomes (``bias = mean(y - m)`` over the
as-of training pairs) — NOT v3's live in-memory bias table, which would leak. The adapter reads no
same-day obs (its leak guard is structural — no ``obs`` argument exists).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, UTC

import numpy as np
import sqlalchemy as sa

from weatherquant.calibrate.link import predict
from weatherquant.calibrate.strata import (
    OBS_SOURCE,
    SIGMA_FLOOR_F,
    assemble_pairs_from_rows,
    fit_pooled_month_strata,
)
from weatherquant.db.models import NATURAL_KEYS, metadata
from weatherquant.db.types import Bind, exec_bind
from weatherquant.price.blend import accuracy_weights, blend_gaussians
from weatherquant.price.buckets import bucket_probs, integers_in_bucket
from weatherquant.price.ticker import TICKER_CITY_SUFFIX_TO_KEY, parse_ticker
from weatherquant.registry import get_city
from weatherquant.verify.v3_reference import v3_bucket_probs

logger = logging.getLogger(__name__)

__all__ = ["PairedRecord", "assert_window_disjoint", "walk_forward"]

#: The pre-registered primary decision lead's cutoff is ``D - PRIMARY_DECISION_LEAD_DAYS`` at
#: 00:00 UTC (a fixed point-in-time before the LST settlement day D, D-05). The walk-forward
#: consumes only rows stamped ``available_at < cutoff``. Conservative (a full day ahead) so the
#: as-of read never peeks at a same-window forecast revision.
PRIMARY_DECISION_LEAD_DAYS: int = 1


@dataclass(frozen=True)
class PairedRecord:
    """One as-of-correct paired observation: WQ vs v3 on the same (day, city, bucket) outcome.

    ``o_i`` is the realized ``{0, 1}`` YES outcome for the bucket. ``excluded_reason`` is ``None``
    for a scored row, or a short reason string for a coverage-logged exclusion (D-09 — never a
    silent drop).

    The per-record predictive parameters (``wq_mu``/``wq_sigma`` for the WQ arm, ``v3_mu``/
    ``v3_sigma`` for the v3 arm, and the realized ``y``) are carried (Optional, default ``None``)
    so Plan 06-07 can score real Gaussian CRPS per record for each arm. They default to ``None``
    so the original six positional fields (and their order) and the existing constructor calls
    stay valid.
    """

    day: object
    city: str
    bucket: object
    wq_prob: float
    v3_prob: float
    o_i: int
    excluded_reason: str | None = None
    wq_mu: float | None = None
    wq_sigma: float | None = None
    v3_mu: float | None = None
    v3_sigma: float | None = None
    y: float | None = None


def assert_window_disjoint(
    test_window: tuple[date, date] | None,
    phase3_oos_slice: tuple[date, date] | None,
) -> None:
    """Fail loud if the Gate-1 test window overlaps the Phase-3 OOS slice (D-10, anti-p-hacking).

    Both windows are half-open ``[start, end)`` date pairs. Raises ``ValueError`` on ANY overlap
    so the Gate-1 proof is never scored on the data used to select calibration hyperparameters
    (D-12 in ``evaluate.py``). A ``None`` window/slice is a no-op (nothing to overlap) — the
    descriptive/integration paths that pass no explicit windows skip the guard cleanly.
    """
    if test_window is None or phase3_oos_slice is None:
        return
    t_start, t_end = test_window
    o_start, o_end = phase3_oos_slice
    # Half-open overlap test: disjoint iff one ends at-or-before the other starts.
    if t_start < o_end and o_start < t_end:
        raise ValueError(
            "Gate-1 test window "
            f"[{t_start}, {t_end}) overlaps the Phase-3 OOS slice [{o_start}, {o_end}); "
            "they MUST be disjoint (D-10/D-12 anti-p-hacking — never score on tuning data)."
        )


def _resolve_city_key(city: str) -> str:
    """Map a Kalshi ``KXHIGH<SUFFIX>`` market code (or a bare registry key) to the registry key.

    The ledger ``city`` column and ``registry.get_city`` use the internal key (e.g. ``NYC``); the
    Gate-1 callers pass the Kalshi market code (e.g. ``KXHIGHNY``). Strips the ``KXHIGH`` prefix
    and maps the suffix via ``price.ticker.TICKER_CITY_SUFFIX_TO_KEY``; a bare registry key passes
    through unchanged so both call styles work.
    """
    if city.startswith("KXHIGH"):
        suffix = city[len("KXHIGH"):]
        return TICKER_CITY_SUFFIX_TO_KEY.get(suffix, city)
    return city


def _decision_cutoff(day: date) -> datetime:
    """The fixed point-in-time decision cutoff BEFORE LST day ``D`` (D-05, no look-ahead).

    ``D - PRIMARY_DECISION_LEAD_DAYS`` at 00:00 UTC. The as-of read keeps only rows with
    ``available_at < cutoff`` — a conservative whole-day lead so a same-window forecast revision
    can never enter the day's training set.
    """
    return datetime(day.year, day.month, day.day, tzinfo=UTC) - timedelta(
        days=PRIMARY_DECISION_LEAD_DAYS
    )


def _latest_as_of(
    bind: Bind, table_name: str, cutoff: datetime, *, where: dict[str, object] | None = None
) -> list[dict]:
    """Point-in-time ``latest`` read: newest row per natural key among ``available_at < cutoff``.

    The as-of analogue of ``db.queries.latest`` — the cutoff is applied INSIDE the SELECT (a
    ``WHERE available_at < :cutoff`` BEFORE the DISTINCT ON), so a future-stamped row is excluded
    from the window rather than filtered after ``latest()`` (RESEARCH §Pitfall 1 — filtering after
    ``latest`` leaks). Keeps the same DISTINCT-ON-natural-key, ``available_at DESC, id DESC``
    ordering as the production reader. Returns plain dicts (detached from the row mapping).
    """
    table = metadata.tables[table_name]
    key_cols = [table.c[name] for name in NATURAL_KEYS[table_name]]
    stmt = (
        sa.select(table)
        .where(table.c.available_at < cutoff)  # POINT-IN-TIME cutoff, applied in the window
        .distinct(*key_cols)
        .order_by(*key_cols, table.c.available_at.desc(), table.c.id.desc())
    )
    if where:
        stmt = stmt.where(*(table.c[name] == value for name, value in where.items()))
    with exec_bind(bind, write=False) as conn:
        return [dict(row) for row in conn.execute(stmt).mappings().all()]


def _outcome_for_bucket(
    y: float, lo: int | None, hi: int | None, open_lo: bool, open_hi: bool
) -> int:
    """Binary YES outcome ``o_i`` for one bucket given the settled daily high ``y`` (RESEARCH).

    Uses the ``price.buckets._HALF`` edge rule (D-05): integer degree ``k`` owns
    ``[k - 0.5, k + 0.5)``. The continuous span is taken from :func:`integers_in_bucket` so the
    outcome geometry is the SAME geometry both arms price (never a re-derived edge). ``y`` is the
    settled ``observations.daily_high_f`` — read ONLY here to score the outcome, never a feature.
    """
    lo_edge, hi_edge = integers_in_bucket(lo, hi, open_lo=open_lo, open_hi=open_hi)
    return 1 if (lo_edge <= y < hi_edge) else 0


def _point_in_time_bias(pairs) -> float:
    """v3 point-in-time bias ``mean(y - m)`` over the as-of training pairs (D-02/A4, leak-safe).

    Mirrors v3's ``derive_bias`` intent (``bias = mean(settled - corrected_mean)``) but measured
    ONLY from ``available_at < cutoff`` settled outcomes — NOT v3's live in-memory bias table. An
    empty training set yields a 0.0 bias (no point-in-time signal yet).
    """
    if not pairs:
        return 0.0
    residuals = np.array([p.y - p.m for p in pairs], dtype=float)
    return float(residuals.mean())


def _v3_arm_raw_ensemble(pairs, day: date, month: int) -> tuple[float, float] | None:
    """The RAW decision-month ensemble pair ``(m_asof, s2_asof)`` for the v3 arm (CR-02-for-v3/D-02).

    verify-subtree D-02 (raw-ensemble v3): the v3 baseline must be priced from the SAME ledger
    ensemble rows the WQ arm sees — the raw ``m`` / ``s2`` off ``assemble_pairs_from_rows`` — NOT
    from WQ's EMOS-corrected ``mu_b`` / Vincentized ``sigma_b``. Otherwise the "v3 baseline" is
    partly WQ's own calibration and the apples-to-apples comparison is voided (CR-05).

    MONTH FILTER FIRST (CR-02-for-v3, GAP 1): the v3 arm is restricted to the decision day's month
    — ``month_pairs = [p for p in pairs if p.month == month]`` — mirroring the WQ arm's CR-02
    month-selection in :func:`_blend_arm_for_day` (which selects the single ``fit.month == month``),
    so BOTH arms price the SAME seasonal subset and methodology is the only remaining difference
    (VER-04). Without this filter the mean averages ``m``/``s2`` across the ENTIRE as-of training
    set — every retained month, all seasons — flattening a July baseline toward the cross-season
    midpoint (the verifier-confirmed contamination: ``v3_mu≈56`` for a July decision day).

    On the PRODUCTION no-look-ahead path the decision-day pair is ABSENT: ``available_at < cutoff``
    filters it out because ``assemble_pairs_from_rows`` only builds a pair when a settled obs exists
    for ``(city, target_date)``, and the daily-high for day D is not known until on/after D. So the
    MONTH-FILTERED mean is the production-NORMAL branch, not a rare fallback. The decision-day
    preference (``p.target_date == day`` → return that pair verbatim) is retained for the rare case
    a decision-day pair is present (e.g. a back-dated obs in a test fixture), but it is selected
    only from ``month_pairs`` so a present cross-season pair can never be picked.

    Returns ``None`` when the decision month has no as-of pairs (absence is absence — the caller
    coverage-logs ``no_v3_ensemble``, mirroring the WQ arm's ``None`` contract in
    :func:`_blend_arm_for_day`). The v3 spread floor ``max(spread, 0.5)`` is applied INSIDE
    ``v3_bucket_probs`` (the legacy contract), so this returns the un-floored raw ``s2``; the
    caller takes ``sqrt(s2)`` as the v3 spread.
    """
    month_pairs = [p for p in pairs if p.month == month]
    if not month_pairs:
        return None
    # Decision-day preference, selected ONLY from the decision-month subset (never cross-season).
    for p in month_pairs:
        if p.target_date == day:
            return float(p.m), float(p.s2)
    # Production-normal: the decision-MONTH mean ensemble (point-in-time, never a future row).
    m_asof = float(np.mean([p.m for p in month_pairs]))
    s2_asof = float(np.mean([p.s2 for p in month_pairs]))
    return m_asof, s2_asof


def _blend_arm_for_day(pairs, *, city_key: str, model: str, lead: int, month: int):
    """Refit EMOS and return the DECISION DAY'S month-fit WQ blend ``(mu_b, sigma_b)`` (D-04/CR-02).

    PURE REUSE (no new calibration math, verify-subtree D-04): ``fit_pooled_month_strata`` (the
    existing Phase-3 fitter) → per-month ``StratumFit``; this selects the SINGLE fit whose
    ``fit.month == month`` (the decision day D's month) and reconstructs ITS predictive Gaussian
    via ``link.predict`` over its own ``StratumSamples``. The blend is across MODELS only — with a
    single present model that is the identity, so the production single-model blend path is
    preserved verbatim. It NEVER iterates all retained month-fits into an equal-weight
    ``blend_gaussians`` average across the twelve calendar months: a July day must be priced from
    the July fit, not a cross-month midpoint (CR-02 — the seasonal-contamination defect).

    Returns ``None`` when no retained month-fit covers ``month`` (the caller coverage-logs
    ``no_month_fit``, D-09 — absence is absence, never silently blended from other months).
    """
    fits = fit_pooled_month_strata(pairs, city=city_key, model=model, lead=lead)
    # CR-02: select the ONE fit for the decision day's month — never an across-month average.
    selected = next(
        ((samples, fit) for samples, _target_dates, fit in fits if fit.month == month),
        None,
    )
    if selected is None:
        return None
    month_samples, fit = selected
    params = (fit.a, fit.b, fit.c, fit.d, fit.sigma_floor)
    mu, sigma = predict(params, month_samples.m, month_samples.s2)
    # One representative single-model component: the mean over the fit's own training rows (its
    # central predictive Gaussian). The blend over a single present model is the identity, mirroring
    # the production single-model blend (D-01) — accuracy weight degrades to 1.0 for one component.
    mus = np.array([float(np.mean(mu))], dtype=float)
    sigmas = np.array([float(np.mean(sigma))], dtype=float)
    weights = accuracy_weights(np.array([1.0], dtype=float))
    mu_b, sigma_b = blend_gaussians(mus, sigmas, weights)
    sigma_b = max(sigma_b, SIGMA_FLOOR_F)  # never a degenerate sigma into the CDF (D-09)
    return mu_b, sigma_b


def _settlement_days(bind: Bind, city_col: str, start: date | None, end: date | None) -> list[date]:
    """The ordered list of LST settlement days to replay over ``[start, end)`` (stable, D-10).

    When ``start``/``end`` are given, replays every calendar day in the half-open window. When
    they are ``None`` (the descriptive/integration path), derives the window from the distinct
    ``observations.target_date`` rows present in the ledger for the city (empty ⇒ no days). Days
    are always returned in ascending date order (the temporal-split ordering discipline; never
    shuffled — RESEARCH §Pattern 5 / ``evaluate.temporal_split``).
    """
    if start is not None and end is not None:
        days: list[date] = []
        d = start
        while d < end:
            days.append(d)
            d = d + timedelta(days=1)
        return days
    # No explicit window: enumerate the settlement days the ledger actually has obs for.
    obs = metadata.tables["observations"]
    stmt = (
        sa.select(obs.c.target_date)
        .where(obs.c.city == city_col, obs.c.source == OBS_SOURCE)
        .distinct()
        .order_by(obs.c.target_date.asc())
    )
    with exec_bind(bind, write=False) as conn:
        rows = conn.execute(stmt).all()
    return [r[0] for r in rows]


def walk_forward(
    bind: Bind | None,
    city: str,
    model: str,
    lead: int,
    start: date | None,
    end: date | None,
    oos_slice: tuple[date, date] | None,
):
    """Replay the ledger as-of-correctly and assemble paired WQ-vs-v3 records (VER-03, D-08).

    For each LST settlement day D in ``[start, end)`` (or every ledger-present obs day when the
    window is ``None``): assert the test window is disjoint from the Phase-3 ``oos_slice`` (D-10,
    done FIRST), read ONLY ledger rows with ``available_at < _decision_cutoff(D)`` (no look-ahead,
    T-06-12), refit EMOS via the existing Phase-3 fitter (D-04 — no new math), price BOTH arms on
    the IDENTICAL bucket geometry (WQ via ``price.blend``→``price.buckets``; v3 via
    ``verify.v3_reference`` with a point-in-time bias, D-02/A4), and settle each bucket against the
    observed ``daily_high_f`` (read ONLY to score ``o_i``, never a feature).

    Returns ``(records, coverage_log)``: ``records`` is the list of scored
    :class:`PairedRecord` (one per day×bucket at the primary lead); ``coverage_log`` is the list
    of ``{day, city, reason}`` exclusion dicts — every voided/missing/non-settling day is logged
    with a reason (D-09), never silently dropped. The disjointness guard runs before any ledger
    access so an overlapping window fails loud even with ``bind=None``.
    """
    # D-10 disjointness FIRST — before any scoring or ledger access (fails loud on overlap).
    assert_window_disjoint((start, end) if start and end else None, oos_slice)

    records: list[PairedRecord] = []
    coverage_log: list[dict] = []

    if bind is None:
        # No ledger to replay (disjointness-only call path) — nothing to score.
        return records, coverage_log

    city_key = _resolve_city_key(city)
    try:
        get_city(city_key)  # validate the city resolves (its window clock is settlement_window)
    except KeyError:
        logger.warning(
            "walk_forward: unknown city %r (resolved key %r) — no settlement window; "
            "logging full-coverage exclusion",
            city,
            city_key,
        )
        coverage_log.append({"day": None, "city": city, "reason": "unknown_city"})
        return records, coverage_log

    for day in _settlement_days(bind, city_key, start, end):
        cutoff = _decision_cutoff(day)

        # --- POINT-IN-TIME as-of read (available_at < cutoff), applied INSIDE the loop ---------
        forecast_rows = _latest_as_of(
            bind, "forecasts", cutoff, where={"city": city_key, "model": model}
        )
        obs_rows = _latest_as_of(
            bind, "observations", cutoff, where={"city": city_key, "source": OBS_SOURCE}
        )
        pairs = assemble_pairs_from_rows(forecast_rows, obs_rows)
        if not pairs:
            _log_exclusion(coverage_log, day, city, "no_training_data")
            continue

        blend = _blend_arm_for_day(
            pairs, city_key=city_key, model=model, lead=lead, month=day.month
        )
        if blend is None:
            # CR-02/D-09: the decision day's month has no retained month-fit — coverage-log it,
            # never silently blend the WQ arm from other months.
            _log_exclusion(coverage_log, day, city, "no_month_fit")
            continue
        mu_b, sigma_b = blend

        # --- the settled outcome: read daily_high_f ONLY to score o_i (never a feature) --------
        settle_rows = _latest_as_of(
            bind,
            "observations",
            datetime(day.year, day.month, day.day, tzinfo=UTC) + timedelta(days=2),
            where={"city": city_key, "target_date": day, "source": OBS_SOURCE},
        )
        y = _settled_high(settle_rows)
        if y is None:
            _log_exclusion(coverage_log, day, city, "no_settled_outcome")
            continue

        # --- price BOTH arms on the IDENTICAL bucket ladder ------------------------------------
        ladder = _ladder_for_day(mu_b, sigma_b)
        if not ladder:
            _log_exclusion(coverage_log, day, city, "no_market_ladder")
            continue

        # CR-05/D-02: the v3 arm is priced from the RAW decision-day ensemble (m_asof, sqrt(s2_asof))
        # plus the point-in-time bias — INDEPENDENT of WQ's EMOS mu_b / Vincentized sigma_b — so
        # methodology is the only difference (VER-04). The v3 spread floor max(spread, 0.5) is
        # applied inside v3_bucket_probs; do NOT pre-floor with SIGMA_FLOOR_F.
        m_asof, s2_asof = _v3_arm_raw_ensemble(pairs, day)
        bias = _point_in_time_bias(pairs)
        v3_mu = m_asof + bias
        v3_sigma = float(np.sqrt(s2_asof))
        # The shared continuous spans (lo_edge, hi_edge, open_lo, open_hi) — the ONE geometry both
        # arms price; never re-derived per arm (VER-04 — methodology is the only difference).
        wq_ladder = [b["span"] for b in ladder]
        v3_ladder = [b["entry"] for b in ladder]
        # WQ arm: production blend → bucket CDF differencing (price.buckets).
        wq = bucket_probs(mu_b, sigma_b, wq_ladder)
        # v3 arm: RAW ensemble mean/spread + point-in-time bias (CR-05/D-02/A4, leak-safe) — NOT
        # mu_b/sigma_b.
        v3 = v3_bucket_probs(v3_mu, v3_sigma, v3_ladder)
        v3_list = list(v3.values())

        for i, b in enumerate(ladder):
            lo, hi, open_lo, open_hi = b["edges"]
            o_i = _outcome_for_bucket(y, lo, hi, open_lo, open_hi)
            # CR-03/D-09: when the settled high lands in an OPEN tail bucket (its YES), coverage-log
            # the day as tail_settlement so the placement is auditable — the day is STILL scored
            # (the open tail is a real YES, never an o_i=0-everywhere silent drop). Logged once per
            # tail-settled day (on the YES tail bucket), not per interior bucket.
            if o_i == 1 and (open_lo or open_hi):
                _log_exclusion(coverage_log, day, city, "tail_settlement")
            records.append(
                PairedRecord(
                    day=day,
                    city=city,
                    bucket=(lo, hi),
                    wq_prob=float(wq[i]),
                    v3_prob=float(v3_list[i]),
                    o_i=o_i,
                    # Per-record predictive params carried for Plan 06-07 real-CRPS scoring
                    # (CR-02 WQ arm = the decision day's month-fit Gaussian; CR-05 v3 arm = the
                    # raw decision-day ensemble (m+bias, sqrt(s2))).
                    wq_mu=float(mu_b),
                    wq_sigma=float(sigma_b),
                    v3_mu=float(v3_mu),
                    v3_sigma=float(v3_sigma),
                    y=float(y),
                )
            )

    return records, coverage_log


def _log_exclusion(coverage_log: list[dict], day: date, city: str, reason: str) -> None:
    """Append a coverage-log exclusion AND warn (D-09 — never a silent drop on the day axis)."""
    logger.warning("walk_forward exclude day=%s city=%s reason=%s", day, city, reason)
    coverage_log.append({"day": day, "city": city, "reason": reason})


def _settled_high(settle_rows: list[dict]) -> float | None:
    """The settled ``daily_high_f`` from the as-of obs read, or ``None`` if absent/voided."""
    for row in settle_rows:
        y = row.get("daily_high_f")
        if y is not None:
            return float(y)
    return None


def _bucket_geometry(
    *,
    floor_strike: int | None,
    cap_strike: int | None,
    strike_type: str,
) -> dict:
    """Parse ONE Kalshi strike into the shared ``{entry, edges, span}`` geometry (VER-04).

    The single seam both arms price through: ``entry`` is the structured-strike mapping the v3
    adapter parses; ``edges`` is ``(lo, hi, open_lo, open_hi)`` for the outcome rule; ``span`` is
    the continuous ``(lo_edge, hi_edge, open_lo, open_hi)`` (open tails carry the ∓inf sentinel via
    :func:`integers_in_bucket`) the WQ arm's ``bucket_probs`` consumes. No re-derived edge math —
    the interior degree buckets AND the open tails go through the IDENTICAL price-geometry helpers
    so both arms see one ladder.
    """
    entry = {
        "floor_strike": floor_strike,
        "cap_strike": cap_strike,
        "strike_type": strike_type,
    }
    lo_i, hi_i, open_lo, open_hi = parse_ticker(
        floor_strike=floor_strike, cap_strike=cap_strike, strike_type=strike_type
    )
    lo_edge, hi_edge = integers_in_bucket(lo_i, hi_i, open_lo=open_lo, open_hi=open_hi)
    return {
        "entry": entry,
        "edges": (lo_i, hi_i, open_lo, open_hi),
        "span": (lo_edge, hi_edge, open_lo, open_hi),
    }


def _ladder_for_day(mu_b: float, sigma_b: float) -> list[dict]:
    """Build the shared bucket ladder that TILES ``(-inf, +inf)``, as a list of geometry dicts.

    The live KXHIGH market ladder is not persisted in the Gate-1 ledger window. Kalshi's real
    KXHIGH ladder already carries open ``<=lo`` / ``>=hi`` tails, so when it is available in the
    ledger it is PREFERRED (it tiles the line by construction); absent it, this synthesizes the
    same shape — interior single-degree closed buckets centered on the blended mean (±4σ, integer
    degrees) PLUS a ``<= lo_min`` open-lower bucket (``open_lo=True``) and a ``>= hi_max``
    open-upper bucket (``open_hi=True``). Both arms price the IDENTICAL spans (VER-04).

    CR-03 (the silent-tail-drop fix): a CLOSED ±4σ ladder scores any realized daily high outside the
    interior range ``o_i=0`` for EVERY bucket (the true outcome has no YES bucket) with no coverage
    entry — exactly the surprise/tail days that move ROI most. The open tails guarantee EVERY
    realized high lands in exactly one bucket; ``walk_forward`` coverage-logs a tail-settled day as
    ``tail_settlement`` so the placement is auditable (D-09/VER-06) — the day is STILL scored.

    Each entry is parsed ONCE into the shared ``{entry, edges, span}`` geometry via the price
    helpers (:func:`_bucket_geometry`). The open-tail edges use the ``∓inf`` sentinel, consistent
    with ``v3_normal_cdf``'s step/asymptote at the tails and ``price.buckets.bucket_probs``'
    open-tail handling, so the tiled WQ ladder sums to ~1. Empty when sigma is non-finite (caller
    coverage-logs it).
    """
    if not np.isfinite(mu_b) or not np.isfinite(sigma_b) or sigma_b <= 0:
        return []
    center = int(round(mu_b))
    half_span = max(1, int(round(4.0 * sigma_b)))
    lo_min = center - half_span
    hi_max = center + half_span - 1  # the last interior degree (range upper bound is exclusive)
    ladder: list[dict] = []
    # Open-LOWER tail: ``<= lo_min`` (open_lo) — every realized high BELOW the interior lands here.
    ladder.append(
        _bucket_geometry(floor_strike=None, cap_strike=lo_min, strike_type="less")
    )
    # Interior single-degree closed buckets (each integer its own [k-0.5, k+0.5) bucket).
    for lo in range(lo_min + 1, hi_max):
        ladder.append(
            _bucket_geometry(floor_strike=lo, cap_strike=lo, strike_type="between")
        )
    # Open-UPPER tail: ``>= hi_max`` (open_hi) — every realized high ABOVE the interior lands here.
    ladder.append(
        _bucket_geometry(floor_strike=hi_max, cap_strike=None, strike_type="greater")
    )
    return ladder
