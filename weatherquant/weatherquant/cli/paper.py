"""``weatherquant paper`` — REAL live-book midpoint into the money path (D-03/D-04/D-08/D-16).

The market seams (``fetch_snapshot``/``persist_snapshot``/``persist_fill``/``KalshiSigner``/
``clv``) plus ``get_engine``/``get_settings`` are imported into this module's namespace so the
run body resolves them as ``cli.paper.*`` (the seams tests monkeypatch). ``_blend_distribution``
is imported from ``.pricing`` (the shared latest→blend→sufficiency body).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from weatherquant.db.engine import get_engine, get_settings
from weatherquant.db.types import Bind
from weatherquant.market import clv
from weatherquant.market.auth import KalshiSigner
from weatherquant.market.client import fetch_snapshot, run_feed
from weatherquant.market.persist import persist_fill, persist_snapshot
from weatherquant.registry import get_city
from weatherquant.time import settlement_window

from .pricing import _blend_distribution, _price_bucket

logger = logging.getLogger(__name__)

# Paper snapshot persist cadence (D-03/D-04). The run_paper loop persists a market_snapshot at
# most this often (debounced) plus on any material book move. It MUST stay strictly finer than
# the CLV closing window (clv.CLV_WINDOW_MINUTES) so that when a book change falls inside the
# closing window the window holds >= 1 persisted snapshot — keeping Phase-6 CLV DERIVABLE, not
# silently sparse (PAP-04 cadence sufficiency, threat T-05-20). Asserted at run_paper start.
PAPER_SNAPSHOT_CADENCE_SECONDS = 60

# The minimum Kelly stake fraction below which the EV+stake gate declines to place a paper
# order (D-04). One sized position per (market, side) held to settlement — a sub-minimum stake
# is "no edge worth the spread", not a churned micro-order.
PAPER_MIN_STAKE_FRACTION = 1e-4

# PAP-04 cadence-sufficiency invariant (enforced by tests/test_cli.py): the TARGET snapshot
# cadence must stay strictly finer than the CLV closing window so a feed-driven loop honouring
# PAPER_SNAPSHOT_CADENCE_SECONDS keeps the window dense. Both are compile-time module constants
# that cannot drift at runtime, so the relationship is checked once in the suite, not re-asserted
# on every import of this single-shot command (which runs no cadence loop — window density is the
# operator's WR-02 responsibility).


def _reflection_midpoint_cents(book: object) -> float:
    """Derive the live-book midpoint in FLOAT-VALUED CENTS from the REFLECTED best levels.

    Kalshi quotes only bids; the yes ask is reflected as ``100 - best_no_bid`` (cents) via
    :func:`weatherquant.market.reflect.yes_ask_levels` (best/cheapest first). The midpoint is
    ``(best_yes_bid + best_yes_ask) / 2`` in CENTS (the half-cent midpoint, e.g. 50.5 for a
    yes bid 50 / reflected yes ask 51). This is the value PERSISTED as ``market_snapshots.mid``
    — unit-consistent with ``best_yes_bid``/``best_no_bid`` (integer cents) and the fill's
    ``avg_price_cents``, so ``clv.clv_cents`` subtracts with NO conversion. The [0,1] pricing
    value is this midpoint divided by 100 — ``run_paper`` computes it inline as
    ``mid_unit = mid_cents / 100.0`` (3 chars of clear code, no wrapper helper).

    Fails loud (raise) when either side of the book is empty — no two-sided market → no
    derivable midpoint (absence = absence, never a fabricated mid).
    """
    from weatherquant.market import reflect

    # The yes BID top-of-book comes off the book via the ONE bid accessor; the yes ASK is
    # reflected from the no bids via the ONE reflection seam (yes_ask = 100 - no_bid). Both
    # top-of-book derivations live in reflect.py — never read a native ask (there is none), never
    # re-derive the 100 - price reflection, and never re-implement the book accessor / max(prices)
    # inline (IN-03).
    best_yes_bid = reflect.best_bid(book, "yes")
    yes_asks = reflect.yes_ask_levels(book)  # reflected from the no bids, cheapest first
    if best_yes_bid is None or not yes_asks:
        raise SystemExit(
            "paper: book is one-sided (missing a yes bid or a reflected yes ask) — "
            "cannot derive a two-sided midpoint (no fabricated mid)."
        )
    best_yes_ask = yes_asks[0][0]  # cheapest reflected yes ask = 100 - best_no_bid
    return (best_yes_bid + best_yes_ask) / 2.0


def _snapshot_event_time(snapshot: Mapping[str, Any]) -> datetime:
    """Thin CLI wrapper over the single ``clv.snapshot_event_time`` seam (DD-1, D-08).

    The snapshot event-time parse has ONE home (``clv.snapshot_event_time``) — it already
    handles the full key set (``event_time``/``available_at``/``snapshot_for``), so the cli no
    longer drifts on ``available_at``. This wrapper only translates the ``ValueError`` (fail
    loud, never now()) into the CLI's ``SystemExit`` surface; it duplicates no parse body. The
    persisted ``available_at`` is therefore the real observed instant of the book, never the
    wall clock (back-dating destroys Phase-6 no-look-ahead, D-08).
    """
    try:
        return clv.snapshot_event_time(snapshot)
    except ValueError as exc:
        raise SystemExit(f"paper: {exc} — refusing to stamp with now() (D-08).") from exc


def _best_price_size(levels: list[Any]) -> int | None:
    """Sum the sizes at the BEST (max) price in ``levels`` (cents pairs), or ``None`` if empty.

    AGGREGATE size per price before taking the best (WR-07): the raw REST snapshot lists are NOT
    guaranteed collapsed to one entry per price, so if two entries share the best price, taking a
    single max-price level's size would undercount the supporting liquidity and skew the CLV
    supporting-size weight. On the WS ``OrderBook`` each price is already unique (sides are stored
    as ``{price: count}`` dicts), so the per-price aggregation is a harmless no-op there — keeping
    ONE shared helper means single-shot (REST, possibly un-collapsed) and watch (WS) share
    identical supporting-size code.
    """
    by_price: dict[int, int] = {}
    for price, size in levels:
        by_price[int(price)] = by_price.get(int(price), 0) + int(size)
    return by_price[max(by_price)] if by_price else None


def _process_book(
    bind: Bind,
    *,
    book: object,
    event_time: datetime,
    seq: int | None,
    ticker: str,
    blend: dict[str, Any],
    pricing_mod: Any,
    fills_mod: Any,
    reflect_mod: Any,
) -> dict[str, Any]:
    """The shared per-book money tail: midpoint → EV/Kelly gate → taker fill → persist (D-08/D-16).

    PROC-01 / WHY: the single-shot REST path and the Plan-02 watch sink must run the IDENTICAL
    midpoint → gate → fill → persist-snapshot → persist-fill sequence; duplicating this ~160-line
    body would let the two paths drift (divergent volume semantics, a CLV-corrupting unit slip, a
    weakened fail-loud guard). Extracting it once means the watch loop only writes its wrapper.

    The book is read through ONE code path for BOTH shapes: ``_reflection_midpoint_cents`` and
    ``reflect_mod`` route every level read through ``reflect._levels`` (PAP-02), which already
    handles a REST Mapping (``book["yes"]``) AND an attribute ``OrderBook`` (``book.yes``) — so
    the supporting-size reads below use the SAME ``reflect_mod._levels(book, side)`` access for
    both. ``event_time`` and ``seq`` are received as PARAMS so this helper never picks a time
    source itself (D-08: never calls now()) — the caller owns the event-time SOURCE (single-shot
    uses ``_snapshot_event_time``; the watch path uses ``OrderBook.event_time``).
    """
    # The REAL reflection-derived live-book midpoint is kept in TWO units because persistence and
    # pricing need different ones: mid_cents (float-valued CENTS) is PERSISTED as
    # market_snapshots.mid so it is unit-consistent with best_*_bid/avg_price_cents and CLV
    # subtracts with no conversion; mid_unit = mid_cents/100.0 is the [0,1] value pricing needs.
    mid_cents = _reflection_midpoint_cents(book)
    mid_unit = mid_cents / 100.0
    if not (0.0 <= mid_unit <= 1.0):  # the reflected mid must be a valid probability (ASVS V5)
        raise SystemExit(f"paper: reflected midpoint {mid_unit} is not in [0, 1]")

    # Feed the REAL [0,1] mid_unit into the SAME money tail run_price mocks (D-08/D-16 loop
    # closed): _price_bucket shrinks the model prob toward the real mid (p_used), and EV + Kelly
    # size on that shrunk belief. mid_unit (NOT mid_cents) feeds pricing — the path is in [0,1].
    prob, pu, ev, stake = _price_bucket(blend, ticker, mid_unit)

    # best_*_bid are the PRICE columns: the best (highest) bid price on each side (cents), read via
    # the ONE bid accessor (reflect.best_bid, IN-03) so the max(prices) reflection is never
    # re-implemented inline. The yes ASK is the reflection 100 - best_no_bid (reflect.py); these
    # prices back the persisted yes mid.
    best_yes_bid = reflect_mod.best_bid(book, "yes")
    best_no_bid = reflect_mod.best_bid(book, "no")
    # The per-snapshot volume is the liquidity BEHIND the persisted yes mid — the top-of-book
    # two-sided SUPPORTING size min(best_yes_bid_size, best_yes_ask_size). The persisted mid is the
    # yes-side midpoint, so the supporting size is the smaller of the best-yes-bid size and the
    # best-yes-ask size. Because Kalshi quotes only bids and the yes ask is reflected as
    # 100 - best_no_bid carrying the best NO bid's SIZE (reflect.py),
    # best_yes_ask_size == best_no_bid_size — so this is min(best_yes_bid_size, best_no_bid_size).
    #
    # WHY this over a summed two-sided union depth: the union over-weights a snapshot deep on the
    # OPPOSITE (no) side but thin on the yes side — its yes-mid is barely supported yet would carry
    # a large CLV weight, biasing the closing mid toward opposite-side-heavy instants (CORR-MED-3).
    # Narrowing to the supporting top-of-book size weights each mid by the liquidity that genuinely
    # backs THIS mid. The reflection's 100 - price is NOT re-derived: the supporting yes-ask size
    # IS the best-no-bid size by construction. Levels are read through reflect._levels so REST and
    # WS book shapes feed _best_price_size identically (one code path).
    yes_levels = list(reflect_mod._levels(book, "yes"))
    no_levels = list(reflect_mod._levels(book, "no"))
    best_yes_bid_size = _best_price_size(yes_levels)
    best_no_bid_size = _best_price_size(no_levels)
    if best_yes_bid_size is None or best_no_bid_size is None:
        raise SystemExit(
            "paper: book is one-sided — cannot derive the top-of-book supporting size behind "
            "the persisted mid (no fabricated volume)."
        )
    volume = int(min(int(best_yes_bid_size), int(best_no_bid_size)))

    # The market_snapshots natural key is (ticker, snapshot_for). Second-resolution event_time
    # sources mean two distinct book states observed in the same wall-clock second would share an
    # ISO-only snapshot_for; queries.latest's DISTINCT ON (ticker, snapshot_for) would then
    # silently discard one (WR-03). Append the monotonic per-book seq so two same-second states
    # remain distinct natural-key rows the closing window can both weight. The closing-window axis
    # is the event_time/available_at datetime, which the seq suffix does not touch.
    snapshot_for = (
        f"{event_time.isoformat()}#{int(seq)}" if seq is not None else event_time.isoformat()
    )
    # CLOSING-WINDOW AXIS CONTRACT (IN-01): the CLV closing window selects on available_at, and
    # clv.snapshot_event_time prefers event_time/available_at over the snapshot_for ISO string.
    # Here all three derive from the ONE observed instant; enforce their agreement at the write
    # boundary (raise, survives -O) so a future writer that lets them diverge is caught here, not
    # by a silently mis-windowed CLV. (The seq suffix is stripped before comparing the ISO instant.)
    if snapshot_for.split("#", 1)[0] != event_time.isoformat():
        raise RuntimeError(
            "paper: snapshot_for instant disagrees with available_at "
            f"({snapshot_for!r} vs {event_time.isoformat()!r}) — the CLV closing-window axis "
            "would diverge (IN-01)."
        )
    persisted_snapshot_times: list[str] = []
    rc_snap = persist_snapshot(
        bind,
        ticker=ticker,
        snapshot_for=snapshot_for,
        best_yes_bid=best_yes_bid,
        best_no_bid=best_no_bid,
        # Persist CENTS: mid_cents is unit-consistent with best_*_bid/avg_price_cents so CLV
        # subtracts directly. NEVER persist the [0,1] mid_unit here.
        mid=mid_cents,
        volume=volume,
        seq=seq,
        detail={"yes": yes_levels, "no": no_levels},
        available_at=event_time,
    )
    if rc_snap == 1:
        persisted_snapshot_times.append(event_time.isoformat())

    # The positive-EV + minimum-stake gate (D-04): only a positive, sized intent simulates a
    # paper order. One sized position per (market, side), held to settlement — no churn.
    fill_summary: dict[str, Any] | None = None
    if ev > 0.0 and stake >= PAPER_MIN_STAKE_FRACTION:
        # Size the paper order conservatively at 1 contract for the shadow sim (the Gate-1
        # credited path is the taker sweep at achievable liquidity; bankroll→contract count
        # realism is a Phase-6 concern). Real sizing lands when live sizing is enabled (Gate 2).
        want_count = 1
        yes_asks = reflect_mod.yes_ask_levels(book)
        fill = fills_mod.taker_sweep(yes_asks, want_count, event_time=event_time)
        if fill is not None:
            trade_id = f"{ticker}:{snapshot_for}:yes"
            # FAIL LOUD on a taker price that rounds out of the valid 1..99c band. The maker-zero
            # guard in insert_fill (writer.py) only fires for is_maker is True, so a taker fill
            # whose size-weighted average rounds to 0 would otherwise persist price=0 and corrupt
            # CLV as closing_mid - 0 — the exact failure mode the maker guard prevents, on the path
            # it exempts (WR-04). Survives python -O via SystemExit (not assert).
            fill_price_cents = int(round(fill.avg_price_cents))
            if not (1 <= fill_price_cents <= 99):
                raise SystemExit(
                    f"paper: taker fill on {ticker} rounds to price={fill_price_cents}c, outside "
                    f"the valid 1..99c band (avg_price_cents={fill.avg_price_cents!r}) — refusing "
                    "to persist a fabricated 0c/out-of-band fill that would corrupt CLV (WR-04)."
                )
            persist_fill(
                bind,
                ticker=ticker,
                trade_id=trade_id,
                side="yes",
                price=fill_price_cents,
                count=fill.count,
                # Fee on the ACHIEVED size-weighted fill price (in [0,1]), NOT the decision-time
                # mid — so the persisted fee is internally consistent with the persisted price
                # whenever the sweep clears away from the mid (thin/multi-level/partial). Feeing on
                # mid_unit double-counted the slippage into the audited fee (WR-03).
                fee=int(round(pricing_mod.exact_fee(fill.count, fill.avg_price_cents / 100.0) * 100)),
                is_maker=fill.is_maker,
                event_time=fill.event_time,
                bucket_prob=prob,
                ev=ev,
                kelly_stake=stake,
                # Record the decision mid alongside the achieved price + the per-fill slippage
                # (achieved - mid) so an audit of the fills row can distinguish "edge realized"
                # from "edge eaten by the sweep" rather than infer it (WR-05).
                detail={
                    "avg_price_cents": fill.avg_price_cents,
                    "partial": fill.partial,
                    "mid_cents": mid_cents,
                    "slippage_cents": fill.avg_price_cents - mid_cents,
                },
                available_at=fill.event_time,
            )
            fill_summary = {
                "trade_id": trade_id,
                "count": fill.count,
                "avg_price_cents": fill.avg_price_cents,
                "partial": fill.partial,
                "shortfall": fill.shortfall,
            }

    return {
        # The [0,1] pricing midpoint fed into p_used/EV/Kelly; mid_cents is its persisted-unit
        # twin (FLOAT-VALUED CENTS, the unit market_snapshots.mid stores).
        "midpoint": mid_unit,
        "mid_cents": mid_cents,
        "p_used": pu,
        "prob": prob,
        "ev": ev,
        "stake": stake,
        "fill": fill_summary,
        "persisted_snapshot_times": persisted_snapshot_times,
    }


class _WatchSink:
    """The ``on_book`` sink for ``paper --watch``: debounced, material-move-aware persist (D-08).

    WATCH-01 / WHY: ``run_feed`` calls ``on_book(ticker, book)`` after EVERY applied
    snapshot/delta, far more often than the CLV window needs. This sink debounces to at most one
    persist per ``PAPER_SNAPSHOT_CADENCE_SECONDS`` PLUS a forced persist on a MATERIAL book move
    (a change in the best yes/no bid PRICE or its supporting SIZE since the last persist), so the
    closing window stays dense (PAP-04, T-05-20) without churning a row per delta. Each persist
    runs the SHARED ``_process_book`` body so the watch and single-shot paths cannot drift
    (PROC-01).

    Event-time discipline (D-08): the stamp is ``book.event_time`` — the real WS instant carried
    after a delta. The first WS ``orderbook_snapshot`` (seq=1) legitimately carries NO event time
    (``event_time is None``); the sink SKIPS that emit rather than back-dating to ``now()`` (a
    fabricated instant would mis-window CLV). The debounce clock is therefore the WS event time,
    not the wall clock — so a scripted-ts test is deterministic.

    Persist-error surfacing (WR-04 / T-051-09): ``client._emit`` CATCHES a raising sink and only
    logs it so a feed hiccup cannot kill the long-running feed. A hard persist failure must NOT be
    silently swallowed (it would make ``--watch`` look successful while writing nothing), so the
    sink records the first such error on ``self.error``; the loop inspects it AFTER the feed drains
    and re-raises as ``SystemExit``.
    """

    def __init__(
        self,
        bind: Bind,
        *,
        ticker: str,
        blend: dict[str, Any],
        pricing_mod: Any,
        fills_mod: Any,
        reflect_mod: Any,
    ) -> None:
        self._bind = bind
        self._ticker = ticker
        self._blend = blend
        self._pricing = pricing_mod
        self._fills = fills_mod
        self._reflect = reflect_mod
        self._last_persist_at: datetime | None = None
        self._last_top: tuple[int | None, int | None, int | None, int | None] | None = None
        self.persisted_count = 0
        self.last_event_time: datetime | None = None
        # The first hard persist failure observed inside the sink (re-raised by the loop, WR-04).
        self.error: BaseException | None = None

    def _top_of_book(self, book: object) -> tuple[int | None, int | None, int | None, int | None]:
        """The material-move key: best yes/no bid PRICE + its supporting SIZE (via the reflect seam).

        A change in any of (best_yes_bid price, best_no_bid price, best_yes_bid size,
        best_no_bid size) is a MATERIAL move that forces a persist even inside the debounce
        interval — the same top-of-book quantities ``_process_book`` persists (one code path for
        both shapes via ``reflect._levels``), so the move test agrees with the persisted row.
        """
        best_yes_bid = self._reflect.best_bid(book, "yes")
        best_no_bid = self._reflect.best_bid(book, "no")
        yes_size = _best_price_size(list(self._reflect._levels(book, "yes")))
        no_size = _best_price_size(list(self._reflect._levels(book, "no")))
        return (best_yes_bid, best_no_bid, yes_size, no_size)

    def _should_persist(self, event_time: datetime, top: tuple[Any, ...]) -> bool:
        """Persist on the FIRST in-window book, on a material move, or once the debounce elapsed."""
        if self._last_persist_at is None:
            return True
        if top != self._last_top:  # a material book move forces a persist (PAP-04 density)
            return True
        elapsed = (event_time - self._last_persist_at).total_seconds()
        return elapsed >= PAPER_SNAPSHOT_CADENCE_SECONDS

    def __call__(self, ticker: str, book: object) -> None:
        # D-08: the stamp is the WS event time; a book with no delta yet (event_time None) has no
        # derivable instant — SKIP it rather than back-date to now() (the first snapshot before any
        # delta legitimately has none).
        event_time = getattr(book, "event_time", None)
        if event_time is None:
            return
        try:
            top = self._top_of_book(book)
            if not self._should_persist(event_time, top):
                return
            result = _process_book(
                self._bind,
                book=book,
                event_time=event_time,
                seq=getattr(book, "seq", None),
                ticker=self._ticker,
                blend=self._blend,
                pricing_mod=self._pricing,
                fills_mod=self._fills,
                reflect_mod=self._reflect,
            )
        except Exception as exc:  # noqa: BLE001 — record + re-raise via the loop (WR-04, T-051-09)
            # _emit would swallow this; remember the FIRST hard failure so the loop fails loud
            # after the feed drains (a swallowed DB error must never make --watch look successful).
            if self.error is None:
                self.error = exc
            raise
        self._last_persist_at = event_time
        self._last_top = top
        self.last_event_time = event_time
        self.persisted_count += len(result["persisted_snapshot_times"])


async def _run_watch_loop(
    bind: Bind,
    *,
    ticker: str,
    blend: dict[str, Any],
    deadline_seconds: float,
    demo: bool,
    signer: Any,
    pricing_mod: Any,
    fills_mod: Any,
    reflect_mod: Any,
    ws_connect: Any = None,
    max_reconnects: int | None = None,
) -> _WatchSink:
    """Drive ``run_feed(on_book=sink)`` until ``deadline_seconds`` elapses, then return the sink.

    WATCH-02 / WHY: ``run_feed`` runs until the connection drains (tests) or indefinitely
    (production). The watch loop bounds it with ``asyncio.wait_for(..., timeout=deadline_seconds)``
    — the EARLIER of the settlement deadline and the ``--max-duration`` cap, computed by the
    caller. A ``TimeoutError`` (the deadline firing while the feed is still open) is the NORMAL
    terminal condition, not an error. ``ws_connect`` is injectable so tests drive a mock connector
    with no live network; production passes ``None`` → ``run_feed`` defaults to ``websockets``.

    A hard persist failure recorded on the sink (``sink.error``) is re-raised as ``SystemExit``
    here, AFTER the feed drains — ``client._emit`` swallowed it to keep the feed alive, so this
    post-drain inspection is the WR-04 mitigation that makes the failure loud (T-051-09).
    """
    sink = _WatchSink(
        bind,
        ticker=ticker,
        blend=blend,
        pricing_mod=pricing_mod,
        fills_mod=fills_mod,
        reflect_mod=reflect_mod,
    )
    feed = run_feed(
        [ticker],
        signer,
        on_book=sink,
        ws_connect=ws_connect,
        demo=demo,
        max_reconnects=max_reconnects,
    )
    try:
        if max_reconnects is None:
            # UNBOUNDED (production) run: bound it by the wall-clock deadline. A TimeoutError (the
            # deadline firing while the feed is still open) is the NORMAL terminal condition.
            await asyncio.wait_for(feed, timeout=deadline_seconds)
        else:
            # BOUNDED run (a finite scripted stream / tests): run_feed returns on its own after
            # max_reconnects connections drain — let it drain rather than racing a wall-clock
            # timeout (the settlement deadline only bounds the UNBOUNDED production loop, T-051-07).
            await feed
    except (asyncio.TimeoutError, asyncio.CancelledError):
        # The deadline fired while the feed was still open — the NORMAL terminal condition for an
        # unbounded production run (the watch window ended).
        logger.info("paper --watch: deadline reached, terminating feed for %s", ticker)
    if sink.error is not None:
        # WR-04 / T-051-09: a persist failure _emit swallowed to keep the feed alive now fails
        # loud, so a silently-dropped DB write can never make --watch look successful.
        raise SystemExit(
            f"paper --watch: a snapshot/fill persist failed for {ticker} ({sink.error!r}) — "
            "refusing to report success on a swallowed persist error (WR-04)."
        ) from sink.error
    return sink


def run_paper(args: argparse.Namespace) -> dict[str, Any]:
    """Paper-trade one (city, date, ticker): REAL live-book midpoint into the money path (D-08/D-16).

    The Phase-5 loop closer. For the validated ``(city, date, lead, ticker)`` this:

    1. confirms paper mode — there is NO order-submission path anywhere under ``market/`` this
       milestone, so an accidental live order is structurally unreachable (D-15, T-05-14);
    2. pulls a REAL Kalshi orderbook snapshot via :func:`weatherquant.market.client.fetch_snapshot`
       (signed REST; demo host with ``--demo``);
    3. derives the live-book midpoint from the REFLECTED best levels
       (``yes_ask = 100 - best_no_bid`` → ``[0,1]``, the ONE reflection seam) — this is the
       value the loop closes on;
    4. reuses the shared :func:`_blend_distribution` body, then feeds the REAL midpoint (not a
       mock) into ``price.p_used`` → ``price.bucket_ev`` → ``price.stake_fraction`` — closing
       the Phase-4 D-08/D-16 loop;
    5. applies the positive-EV + minimum-stake gate (D-04): one sized position per (market,
       side), held to settlement; a sub-minimum or non-positive-EV intent simulates NO fill;
    6. on a positive intent, simulates a (possibly partial) fill via ``fills.taker_sweep``
       against the reflected asks;
    7. PERSISTS exactly ONE market snapshot per invocation and (on a positive intent) the fill
       via ``market.persist`` — both stamped with the REAL WS event time (D-08). NOTE: this
       command is single-shot; there is NO in-process debounced cadence loop here. Whether the
       CLV closing window holds >= 1 persisted snapshot (PAP-04 cadence sufficiency) is the
       OPERATOR's per-invocation responsibility — run ``paper`` often enough during the closing
       window. ``PAPER_SNAPSHOT_CADENCE_SECONDS`` is the design-time TARGET cadence a future
       feed-driven loop must honour, not a guarantee this single-shot command provides;
    8. places NO real order. Returns a small result dict (``midpoint``/``p_used``/``ev``/
       ``stake``/fill summary/persisted-snapshot event times).

    All pure math stays in :mod:`weatherquant.price` / :mod:`weatherquant.market.fills`; this
    is the I/O edge.
    """
    from weatherquant import price as pricing
    from weatherquant.market import fills, reflect

    city = args.city
    target = args.date
    lead = args.lead
    ticker = args.ticker

    settings = get_settings()
    # The SIMULATOR-only gate: run_paper runs ONLY when execution_mode is NOT 'live' — in live
    # mode the real order flow (Gate 2) would run, not the shadow simulator, so the simulator
    # bows out. This is a distinct check from market.fills.assert_paper_mode, which is the
    # separate, inverse (fail-CLOSED) fence guarding the future order path; it is NOT the check
    # applied here.
    if settings.execution_mode == "live":
        raise SystemExit(
            "paper: execution_mode='live' — the paper-fill simulator does not run in live "
            "mode (no order-submission path exists this milestone, D-15/T-05-14)."
        )

    bind = get_engine()
    signer = KalshiSigner.from_settings(settings)

    import httpx

    from weatherquant.market.client import _resolve_hosts

    # Resolve the COMPLETE (ws_url, rest_host) environment pair from the single --demo flag via
    # the one host-resolution seam (WR-06). Today run_paper does a one-shot REST fetch with no WS,
    # so only rest_host is consumed here — but resolving BOTH from --demo through _resolve_hosts
    # keeps the flag's meaning complete and stable: when a WS feed is wired into run_paper, ws_url
    # is already the matching demo/prod host, never a cross-environment book on the money path.
    demo = bool(getattr(args, "demo", False))
    _, rest_host = _resolve_hosts(demo)

    # The blend is per-(city, date, lead) — NOT per-book — and does not change intra-window, so
    # compute it ONCE before either path (recomputing it per book under --watch would be wasteful).
    blend = _blend_distribution(bind, city, target, lead)

    # --watch: the feed-driven loop (WATCH-01/WATCH-02, un-defers Phase-5's B2). When set, drive
    # market.client.run_feed against the LIVE book instead of the single-shot REST read; the
    # single-shot path below is unchanged when --watch is off.
    if getattr(args, "watch", False):
        # Resolve the City BEFORE computing the deadline: args.city is a validated STR
        # (_args._city_type returns the string, not a City), but time.settlement_window dereferences
        # city.std_offset_hours / city.cli_station — passing the raw arg string would AttributeError
        # (the single-shot path never calls settlement_window, so this is new). get_city is the same
        # resolver _city_type / clv.closing_window_snapshots use.
        deadline = settlement_window(get_city(city), target).end_utc
        max_duration = int(getattr(args, "max_duration", 14400))
        now = datetime.now(UTC)
        # The effective stop is the EARLIER of the settlement deadline and now + max_duration: the
        # duration cap is a SAFETY bound for a missing/garbled boundary, so it must never EXTEND past
        # the settlement deadline (T-051-07). A non-positive remaining window (deadline already past)
        # clamps to 0 so wait_for terminates immediately rather than waiting on a negative timeout.
        deadline_seconds = max(0.0, min((deadline - now).total_seconds(), float(max_duration)))
        # ws_connect is injectable (default None → run_feed uses websockets.connect) so tests drive
        # a mock connector with no live network; max_reconnects likewise (None = run indefinitely).
        ws_connect = getattr(args, "ws_connect", None)
        max_reconnects = getattr(args, "max_reconnects", None)
        sink = asyncio.run(
            _run_watch_loop(
                bind,
                ticker=ticker,
                blend=blend,
                deadline_seconds=deadline_seconds,
                demo=demo,
                signer=signer.sign,
                pricing_mod=pricing,
                fills_mod=fills,
                reflect_mod=reflect,
                ws_connect=ws_connect,
                max_reconnects=max_reconnects,
            )
        )
        logger.info(
            "paper --watch city=%s date=%s ticker=%s deadline=%s persisted=%d last_event=%s",
            city, target, ticker, deadline.isoformat(), sink.persisted_count, sink.last_event_time,
        )
        print(
            f"paper --watch {city} {target} {ticker}: persisted={sink.persisted_count} "
            f"deadline={deadline.isoformat()} "
            f"last_event={'none' if sink.last_event_time is None else sink.last_event_time.isoformat()}"
        )
        return {
            "city": city,
            "date": target.isoformat(),
            "lead": lead,
            "ticker": ticker,
            "models": blend["used_models"],
            "watch": True,
            "deadline": deadline.isoformat(),
            "persisted_snapshot_count": sink.persisted_count,
            "last_event_time": sink.last_event_time,
        }

    async def _fetch() -> dict[str, Any]:
        async with httpx.AsyncClient() as http:
            return await fetch_snapshot(http, signer.sign, ticker, rest_host=rest_host)

    snapshot = asyncio.run(_fetch())
    # The event-time SOURCE stays in the caller (single-shot parses the REST snapshot; the Plan-02
    # watch path will pass OrderBook.event_time). _process_book receives event_time + seq as PARAMS
    # and never picks a source itself (D-08: the helper never calls now()).
    event_time = _snapshot_event_time(snapshot)
    seq = snapshot.get("seq")

    # Persist exactly ONE snapshot for this invocation (this command is single-shot; there is no
    # in-process cadence loop, WR-02). DELEGATE the per-book money tail (midpoint → EV/Kelly gate →
    # taker fill → persist snapshot + fill) to the shared _process_book helper so the Plan-02 watch
    # sink runs the IDENTICAL body — no drift in volume/event-time/fail-loud semantics (PROC-01).
    result = _process_book(
        bind,
        book=snapshot,
        event_time=event_time,
        seq=seq,
        ticker=ticker,
        blend=blend,
        pricing_mod=pricing,
        fills_mod=fills,
        reflect_mod=reflect,
    )

    logger.info(
        "paper city=%s date=%s ticker=%s midpoint=%.4f mid_cents=%.2f p_used=%.4f ev=%+.4f "
        "stake=%.4f fill=%s",
        city, target, ticker,
        result["midpoint"], result["mid_cents"], result["p_used"], result["ev"], result["stake"],
        result["fill"],
    )
    print(
        f"paper {city} {target} {ticker}: mid={result['midpoint']:.4f} ({result['mid_cents']:.2f}c) "
        f"P={result['prob']:.4f} EV={result['ev']:+.4f} stake={result['stake']:.4f} "
        f"fill={'none' if result['fill'] is None else result['fill']['count']}"
    )

    return {
        "city": city,
        "date": target.isoformat(),
        "lead": lead,
        "ticker": ticker,
        "models": blend["used_models"],
        **result,
    }
