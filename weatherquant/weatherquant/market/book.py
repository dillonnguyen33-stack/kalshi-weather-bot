"""In-memory per-ticker orderbook + seq integrity (PAP-01, D-02).

The book is rebuilt from a Kalshi V2 ``orderbook_snapshot`` (a full reset to a ``seq``), then
mutated by contiguous ``orderbook_delta`` messages. Each delta MUST carry ``book.seq + 1``;
any non-contiguous ``seq`` (a gap) raises :class:`SeqGap` — the book is then treated as
UNKNOWN, never carried forward silently (D-02, threat T-05-08). The book-level raise is all
this module owns; the gap-RECOVERY mechanism (re-subscribe for a fresh WS snapshot) lives in
:mod:`weatherquant.market.client` (05-12).

VERIFIED LIVE SCHEMA (captured 2026-06-19; 05-UAT.md ## Gaps → verified_live_schema):

* CONTROL FRAME: ``{"type": "subscribed", "id": 1, "msg": {...}}`` arrives first after a
  subscribe (and the documented siblings ``ok`` / ``error`` / ``unsubscribed``). It carries no
  book data and is IGNORED — never fail-loud (the prior flat design crashed on it).
* ``orderbook_snapshot``: an ENVELOPE — top-level ``type`` / ``sid`` / ``seq`` with the book
  data nested under ``msg``: ``msg.yes_dollars_fp`` / ``msg.no_dollars_fp`` are
  ``[[price_dollar_string, count_fixedpoint_string], ...]`` BID levels (no native asks; the ask
  is reflected as ``100 - opposite bid`` in :mod:`weatherquant.market.reflect`). The
  per-subscription ``seq`` (snapshot = 1) is the integrity ANCHOR — the WS snapshot, NOT the
  REST orderbook (which carries no seq at all), anchors the book (D-02 revised; the
  decision-record edit is 05-12's, not this plan's).
* ``orderbook_delta``: an ENVELOPE — ``msg.side`` / ``msg.price_dollars`` / ``msg.delta_fp`` is
  a single signed size change at one ``(side, price_cents)`` level, and ``msg.ts`` (ISO-8601
  UTC) / ``msg.ts_ms`` (epoch ms) is the real WS event time carried onto
  :attr:`OrderBook.event_time` (PAP-03; the WS→persistence wiring is deferred to 05-12).

Units: a price dollar-string × 100 → integer cents; a count/delta fixed-point string →
``round(float(...))`` signed contract count. The internal book is uniformly integer CENTS so
the WS and REST (dollars→cents) encodings converge to ONE representation.

Defensive parsing (ASVS V5, threat T-05-10): a ``type`` that is neither a known data type NOR
a recognized control frame FAILS LOUD; a missing required ``msg`` field, a missing top-level
``seq``, a malformed level, or a ``delta`` before any ``snapshot`` raises — a malformed message
is never coerced into a fabricated level (absence = absence). The ticker key (under ``msg``) is
accepted as ``market_ticker`` OR ``market_id`` OR ``ticker`` (A1); a present-but-non-str ticker
fails loud (TS-1).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from weatherquant.ingest.errors import CorrectnessError

# The documented data message types (verified_live_schema).
_TYPE_SNAPSHOT = "orderbook_snapshot"
_TYPE_DELTA = "orderbook_delta"

# The control-frame types Kalshi sends on the channel (acknowledgements, no book data). The
# SINGLE home for this set so 05-12's run_feed control-frame filter imports the SAME constant
# (one source of truth — W1). ``subscribed`` is the verified first frame; the siblings are the
# documented acknowledgement/teardown frames that also carry no book data.
CONTROL_FRAME_TYPES = frozenset({"subscribed", "ok", "error", "unsubscribed"})

# The two book sides (Kalshi quotes the BID side of each outcome).
_SIDES = ("yes", "no")

# The ticker key is tolerated under every documented spelling (A1), now read from the msg body.
_TICKER_KEYS = ("market_ticker", "market_id", "ticker")

# Dollars → cents: a Kalshi dollar price-string × 100 is the integer-cent price.
_DOLLARS_TO_CENTS = 100


class SeqGap(CorrectnessError, ValueError):
    """A non-contiguous ``seq`` was observed — the local book is UNKNOWN (D-02, T-05-08).

    Subclasses :class:`~weatherquant.ingest.errors.CorrectnessError` so the client's
    correctness-catch re-anchors rather than silently applying an out-of-order delta, and
    :class:`ValueError` so a direct ``pytest.raises(ValueError)`` still holds (mirrors the
    ingest errors' dual base). Carries the ``(expected, got)`` seqs for diagnosis.
    """

    def __init__(self, expected: int, got: int) -> None:
        self.expected = expected
        self.got = got
        super().__init__(
            f"orderbook seq gap: expected {expected} (last seq + 1) but got {got} — "
            "the local book is unknown and must be re-anchored (D-02), never carried "
            "forward silently."
        )


def _cents(dollar_string: str | float | int) -> int:
    """Convert a dollar price-string to integer cents (``round(float(p)*100)``).

    The ONE dollars→cents conversion in this module, mirroring ``client._cents`` exactly so the
    WS and REST encodings converge to the same integer-cent representation.
    """
    return round(float(dollar_string) * _DOLLARS_TO_CENTS)


def parse_dollar_fp_side(raw: Any) -> list[list[int]]:
    """Parse one dollar/fixed-point side (``[[price_dollars, count_fp], ...]``) to cent levels.

    The PRICE is a dollar-string → integer cents via :func:`_cents`; the COUNT is a fixed-point
    string contract count → ``round(float(count))`` (it is NOT a dollar amount, so it is never
    ×100 — that would inflate every resting size). Fails loud on a malformed level (never coerce
    a fabricated level; ASVS V5, T-05-10), reusing the same 2-element level guard as
    ``client._parse_fp_side``. A missing side (``None``) is an empty book side (absence).

    Single source of the dollar/fp side parse; ``client._parse_fp_side`` is a duplicate to be
    DELETED in 05-12 (W1) — that plan imports THIS function and deletes its own copy.
    """
    if raw is None:
        return []
    levels: list[list[int]] = []
    for level in raw:
        # Validate a 2-element [price, count] sequence BEFORE unpacking (mirror the
        # client._parse_fp_side / _coerce_levels len-2 guard): a malformed level raises a
        # descriptive ValueError, never a coerced level or an opaque unpack error (HIGH-2/V5).
        if isinstance(level, (str, bytes, Mapping)):
            raise ValueError(f"orderbook level must be [price, count], got {level!r}")
        items = list(level)
        if len(items) != 2:
            raise ValueError(f"orderbook level must be exactly [price, count], got {items!r}")
        price_d, count = items
        levels.append([_cents(price_d), round(float(count))])
    return levels


def _coerce_levels(raw: Any) -> dict[int, int]:
    """Coerce a parsed snapshot side (cent levels) to a ``{price: count}`` map.

    Drops non-positive counts (a level with no size is absence, not a fabricated level).
    """
    levels: dict[int, int] = {}
    for price, count in raw:
        if count > 0:
            levels[int(price)] = int(count)
    return levels


def _parse_event_time(msg: Mapping[str, Any]) -> datetime | None:
    """Parse a message's WS event time to a tz-aware UTC datetime (PAP-03), or None if absent.

    The ONE event-time parse seam for the book: prefer ``msg.ts`` (ISO-8601 UTC string,
    accepting a trailing ``Z``); fall back to ``msg.ts_ms`` (epoch milliseconds). A naive
    datetime is assumed UTC. An ABSENT event time returns None (tolerated — the caller leaves
    :attr:`OrderBook.event_time` unchanged; never back-date to ``now()``, D-08). A
    present-but-MALFORMED ts/ts_ms FAILS LOUD (never fabricate an instant).
    """
    ts = msg.get("ts")
    if ts is not None:
        if not isinstance(ts, str):
            raise ValueError(f"orderbook msg.ts must be an ISO-8601 string, got {ts!r}")
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)
    ts_ms = msg.get("ts_ms")
    if ts_ms is not None:
        return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
    return None


class OrderBook:
    """An in-memory per-ticker book: a ``seq`` baseline + a ``{price_cents: count}`` map per side.

    The two sides are stored internally as ``{price_cents: count}`` maps (``_yes`` / ``_no``)
    for O(1) delta application, but EXPOSED as the :attr:`yes` / :attr:`no` bid-level lists of
    ``(price_cents, count)`` pairs the reflection seam (:mod:`weatherquant.market.reflect`)
    iterates over — so ``yes_ask_levels(book)`` / ``no_ask_levels(book)`` work directly on this
    in-memory book exactly as they do on the dict fixtures (05-01 D: the dict-or-attribute
    accessor). Kalshi quotes only BIDS; the ask is reflected as ``100 - opposite bid``.

    :attr:`event_time` carries the WS event time (``msg.ts``/``msg.ts_ms``) of the last applied
    delta as a tz-aware UTC datetime (PAP-03); it is None until the first delta carrying a usable
    time (a snapshot envelope may not carry one — absent is fine, never back-dated).

    A fresh ``OrderBook()`` is UNINITIALIZED (``seq is None``) — a delta arriving before any
    snapshot raises (fail loud, never coerce). :func:`apply` is the single mutation entry
    point; callers do not poke the maps directly.
    """

    __slots__ = ("ticker", "seq", "event_time", "_yes", "_no")

    def __init__(self, ticker: str | None = None) -> None:
        self.ticker: str | None = ticker
        self.seq: int | None = None
        self.event_time: datetime | None = None
        self._yes: dict[int, int] = {}
        self._no: dict[int, int] = {}

    @property
    def initialized(self) -> bool:
        """True once a snapshot has reset the book to a ``seq`` baseline."""
        return self.seq is not None

    @property
    def yes(self) -> list[tuple[int, int]]:
        """The ``yes`` BID levels as ``(price_cents, count)`` pairs (reflect-seam shape)."""
        return list(self._yes.items())

    @property
    def no(self) -> list[tuple[int, int]]:
        """The ``no`` BID levels as ``(price_cents, count)`` pairs (reflect-seam shape)."""
        return list(self._no.items())

    def reset(self, seq: int, *, yes: Any, no: Any, ticker: str | None = None) -> None:
        """Full reset to a snapshot: replace ``seq`` and both side maps from parsed cent levels."""
        self.seq = int(seq)
        self._yes = _coerce_levels(yes)
        self._no = _coerce_levels(no)
        if ticker is not None:
            self.ticker = ticker

    def apply_delta(self, *, side: str, price: int, delta: int) -> None:
        """Mutate one level by a signed ``delta``; a level total <= 0 is dropped.

        Does NOT advance ``seq`` — :func:`apply` advances it after the gap check so the seq
        baseline only moves when a contiguous delta is actually applied.
        """
        if side not in _SIDES:
            raise ValueError(f"orderbook delta side must be one of {_SIDES}, got {side!r}")
        levels = self._yes if side == "yes" else self._no
        new_total = levels.get(int(price), 0) + int(delta)
        if new_total > 0:
            levels[int(price)] = new_total
        else:
            # Total at or below zero → the level is gone (absence = absence, never negative).
            levels.pop(int(price), None)


def _msg_get(msg: Mapping[str, Any], key: str) -> Any:
    """Fail-loud accessor for a required message field (ASVS V5, never a silent default)."""
    if key not in msg:
        raise ValueError(f"orderbook message missing required field {key!r}: {msg!r}")
    return msg[key]


def _body_of(msg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Unwrap the enveloped ``msg`` body, fail-loud on a missing/non-Mapping body (V5)."""
    body = _msg_get(msg, "msg")
    if not isinstance(body, Mapping):
        raise ValueError(f"orderbook 'msg' body must be a mapping, got {body!r}")
    return body


def _ticker_of(body: Mapping[str, Any]) -> str | None:
    """Return the ticker under whichever documented key is present in the unwrapped msg body.

    The ticker value is UNTRUSTED WS JSON; a present-but-non-str value FAILS LOUD so the
    declared ``str | None`` is actually enforced (TS-1) rather than coercing a non-str into the
    book. An ABSENT ticker key still returns None — A1 tolerates a snapshot that omits the key
    (absence is the tolerated case; only a present-but-malformed value raises). The ticker now
    lives UNDER the envelope's ``msg`` body, so callers pass the unwrapped body here.
    """
    for key in _TICKER_KEYS:
        if key in body:
            value = body[key]
            if not isinstance(value, str):
                raise ValueError(f"orderbook ticker under {key!r} is not a str: {value!r}")
            return value
    return None


def apply(book: OrderBook, msg: Mapping[str, Any]) -> None:
    """Apply one Kalshi V2 WS message (control / snapshot / delta) to ``book`` (verified schema).

    * A control frame (``type`` in :data:`CONTROL_FRAME_TYPES`) → IGNORED, mutates nothing.
    * ``orderbook_snapshot`` → full reset of ``(seq, yes, no)`` from the envelope: the top-level
      ``seq`` is the WS anchor; the book data is parsed from ``msg.yes_dollars_fp`` /
      ``msg.no_dollars_fp`` dollar/fp strings; the ticker is read from the unwrapped body.
    * ``orderbook_delta`` with ``seq == book.seq + 1`` → mutate the named
      ``(msg.side, msg.price_dollars→cents)`` by the signed ``msg.delta_fp`` (drop a level at
      total 0), advance ``book.seq``, and carry ``msg.ts``/``msg.ts_ms`` onto
      :attr:`OrderBook.event_time` (PAP-03).
    * ``orderbook_delta`` with a NON-contiguous ``seq`` → raise :class:`SeqGap` (D-02).
    * A ``delta`` before any ``snapshot`` (uninitialized book) → raise (fail loud).
    * An unknown ``type`` that is neither a data type nor a control frame, or a missing required
      field → raise (ASVS V5, never coerce, T-05-10).

    The book is mutated in place.
    """
    if not isinstance(msg, Mapping):
        raise ValueError(f"orderbook message must be a mapping, got {msg!r}")
    msg_type = _msg_get(msg, "type")

    # Control frames carry no book data — ignore them, never fail loud (the prior flat design
    # crashed on the very first 'subscribed'). One home for the set (W1).
    if msg_type in CONTROL_FRAME_TYPES:
        return

    if msg_type == _TYPE_SNAPSHOT:
        body = _body_of(msg)
        book.reset(
            int(_msg_get(msg, "seq")),  # WS snapshot is the per-subscription seq anchor (D-02).
            yes=parse_dollar_fp_side(body.get("yes_dollars_fp")),
            no=parse_dollar_fp_side(body.get("no_dollars_fp")),
            ticker=_ticker_of(body),
        )
        # A snapshot envelope MAY carry a ts/ts_ms; carry it if present, else leave unchanged
        # (do not back-date — the first real delta supplies the event time, D-08).
        observed = _parse_event_time(body)
        if observed is not None:
            book.event_time = observed
        return

    if msg_type == _TYPE_DELTA:
        if not book.initialized:
            raise ValueError(
                "orderbook_delta arrived before any orderbook_snapshot — the book has no "
                "seq baseline; fail loud rather than coerce a fabricated level (D-02, V5)."
            )
        got_seq = int(_msg_get(msg, "seq"))
        assert book.seq is not None  # narrowed by .initialized
        expected = book.seq + 1
        if got_seq != expected:
            raise SeqGap(expected, got_seq)
        body = _body_of(msg)
        book.apply_delta(
            side=_msg_get(body, "side"),
            price=_cents(_msg_get(body, "price_dollars")),
            delta=round(float(_msg_get(body, "delta_fp"))),
        )
        book.seq = got_seq
        # Carry the real WS event time onto the book (PAP-03). Absent → leave unchanged (never
        # fabricate now()); present-but-malformed → fail loud inside _parse_event_time.
        observed = _parse_event_time(body)
        if observed is not None:
            book.event_time = observed
        return

    raise ValueError(
        f"unknown orderbook message type {msg_type!r} — expected "
        f"{_TYPE_SNAPSHOT!r}, {_TYPE_DELTA!r}, or a control frame "
        f"{sorted(CONTROL_FRAME_TYPES)} (fail loud, never coerce; T-05-10)."
    )


__all__ = [
    "OrderBook",
    "SeqGap",
    "apply",
    "parse_dollar_fp_side",
    "CONTROL_FRAME_TYPES",
]
