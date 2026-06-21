"""In-memory per-ticker orderbook + seq integrity (PAP-01, D-02).

The book is reset from a Kalshi V2 ``orderbook_snapshot`` then mutated by contiguous
``orderbook_delta`` messages; a non-contiguous ``seq`` raises :class:`SeqGap` and the book is
UNKNOWN, never carried forward (D-02, T-05-08). Gap RECOVERY (re-subscribe for a fresh WS
snapshot) lives in :mod:`weatherquant.market.client`.

Verified live schema (captured 2026-06-19; see docs/DECISIONS.md / 05-UAT.md):

* CONTROL FRAME ``{"type": "subscribed", ...}`` (and siblings ``ok`` / ``error`` /
  ``unsubscribed``): no book data, IGNORED.
* ``orderbook_snapshot``: an ENVELOPE — top-level ``type`` / ``sid`` / ``seq``, book data under
  ``msg.yes_dollars_fp`` / ``msg.no_dollars_fp`` as ``[[price_dollars, count_fp], ...]`` BID
  levels. The per-subscription ``seq`` (snapshot = 1) is the integrity ANCHOR — the WS snapshot,
  not the seq-less REST orderbook (D-02 revised).
* ``orderbook_delta``: an ENVELOPE — ``msg.side`` / ``msg.price_dollars`` / ``msg.delta_fp`` is
  one signed size change at a level, and ``msg.ts`` / ``msg.ts_ms`` is the WS event time carried
  onto :attr:`OrderBook.event_time` (PAP-03).

Units: price dollar-string × 100 → integer cents; count/delta fp string → ``round(float(...))``.
The internal book is uniformly integer CENTS so WS and REST encodings converge.

Defensive parsing (ASVS V5, T-05-10): an unknown ``type``, a missing required ``msg`` field or
top-level ``seq``, a malformed level, or a ``delta`` before any ``snapshot`` FAILS LOUD — never
coerced into a fabricated level. The ticker key is ``market_ticker`` / ``market_id`` / ``ticker``
(A1); a present-but-non-str ticker fails loud (TS-1).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, UTC
from typing import Any

from weatherquant.ingest.errors import CorrectnessError
from weatherquant.time import parse_utc

# The documented data message types.
_TYPE_SNAPSHOT = "orderbook_snapshot"
_TYPE_DELTA = "orderbook_delta"

# Control-frame types (acknowledgements, no book data). The SINGLE home for this set so
# run_feed's filter imports the SAME constant (W1).
CONTROL_FRAME_TYPES = frozenset({"subscribed", "ok", "error", "unsubscribed"})

# The two book sides (Kalshi quotes the BID side of each outcome).
_SIDES = ("yes", "no")

# Tolerated ticker-key spellings (A1), read from the msg body.
_TICKER_KEYS = ("market_ticker", "market_id", "ticker")

# Dollars → cents.
_DOLLARS_TO_CENTS = 100


class SeqGap(CorrectnessError, ValueError):
    """A non-contiguous ``seq`` was observed — the local book is UNKNOWN (D-02, T-05-08).

    Dual base: :class:`CorrectnessError` (so the client re-anchors) and :class:`ValueError`
    (so ``pytest.raises(ValueError)`` holds). Carries the ``(expected, got)`` seqs.
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

    The ONE dollars→cents conversion shared by the WS and REST encodings (W1).
    """
    return round(float(dollar_string) * _DOLLARS_TO_CENTS)


def parse_dollar_fp_side(raw: Any) -> list[list[int]]:
    """Parse one dollar/fixed-point side (``[[price_dollars, count_fp], ...]``) to cent levels.

    PRICE → integer cents via :func:`_cents`; COUNT → ``round(float(count))`` (a contract count,
    never ×100). Fails loud on a malformed level (ASVS V5, T-05-10); a missing side (``None``)
    is an empty side. The single source of the dollar/fp side parse (W1).
    """
    if raw is None:
        return []
    levels: list[list[int]] = []
    for level in raw:
        # Validate a 2-element [price, count] sequence BEFORE unpacking: a malformed level
        # raises a descriptive ValueError, never a coerced level (HIGH-2/V5).
        if isinstance(level, (str, bytes, Mapping)):
            raise ValueError(f"orderbook level must be [price, count], got {level!r}")
        items = list(level)
        if len(items) != 2:
            raise ValueError(f"orderbook level must be exactly [price, count], got {items!r}")
        price_d, count = items
        levels.append([_cents(price_d), round(float(count))])
    return levels


def _coerce_levels(raw: Any) -> dict[int, int]:
    """Coerce a parsed snapshot side (cent levels) to a ``{price: count}`` map, dropping
    non-positive counts."""
    levels: dict[int, int] = {}
    for price, count in raw:
        if count > 0:
            levels[int(price)] = int(count)
    return levels


def _parse_event_time(msg: Mapping[str, Any]) -> datetime | None:
    """Parse a message's WS event time to a tz-aware UTC datetime (PAP-03), or None if absent.

    Prefers ``msg.ts`` (ISO-8601, trailing ``Z`` accepted), falls back to ``msg.ts_ms`` (epoch
    ms); naive datetimes are UTC. Absent → None (caller leaves :attr:`OrderBook.event_time`
    unchanged, never back-dated, D-08); present-but-malformed FAILS LOUD.
    """
    ts = msg.get("ts")
    if ts is not None:
        if not isinstance(ts, str):
            raise ValueError(f"orderbook msg.ts must be an ISO-8601 string, got {ts!r}")
        return parse_utc(ts)
    ts_ms = msg.get("ts_ms")
    if ts_ms is not None:
        return datetime.fromtimestamp(int(ts_ms) / 1000, tz=UTC)
    return None


class OrderBook:
    """An in-memory per-ticker book: a ``seq`` baseline + a ``{price_cents: count}`` map per side.

    Sides are stored as ``{price_cents: count}`` maps (``_yes`` / ``_no``) for O(1) delta
    application but EXPOSED as the :attr:`yes` / :attr:`no` bid-level lists the reflection seam
    iterates over (dict-or-attribute accessor). :attr:`event_time` carries the last applied
    delta's WS event time (PAP-03), None until the first delta with a usable time. A fresh
    ``OrderBook()`` is UNINITIALIZED (``seq is None``); a delta before any snapshot raises.
    :func:`apply` is the single mutation entry point.
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

        Does NOT advance ``seq`` — :func:`apply` advances it after the gap check.
        """
        if side not in _SIDES:
            raise ValueError(f"orderbook delta side must be one of {_SIDES}, got {side!r}")
        levels = self._yes if side == "yes" else self._no
        new_total = levels.get(int(price), 0) + int(delta)
        if new_total > 0:
            levels[int(price)] = new_total
        else:
            # Total <= 0 → the level is gone (never negative).
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

    A present-but-non-str value FAILS LOUD (TS-1); an absent key returns None (A1 tolerates a
    snapshot that omits it). The ticker lives under the envelope's ``msg`` body.
    """
    for key in _TICKER_KEYS:
        if key in body:
            value = body[key]
            if not isinstance(value, str):
                raise ValueError(f"orderbook ticker under {key!r} is not a str: {value!r}")
            return value
    return None


def apply(book: OrderBook, msg: Mapping[str, Any]) -> None:
    """Apply one Kalshi V2 WS message (control / snapshot / delta) to ``book`` in place.

    * control frame (``type`` in :data:`CONTROL_FRAME_TYPES`) → IGNORED.
    * ``orderbook_snapshot`` → full reset of ``(seq, yes, no)`` from the envelope.
    * ``orderbook_delta`` with ``seq == book.seq + 1`` → mutate the named level, advance
      ``book.seq``, carry ``msg.ts``/``msg.ts_ms`` onto :attr:`OrderBook.event_time` (PAP-03).
    * ``orderbook_delta`` with a NON-contiguous ``seq`` → raise :class:`SeqGap` (D-02).
    * a ``delta`` before any ``snapshot``, an unknown ``type``, or a missing required field →
      raise (ASVS V5, T-05-10).
    """
    if not isinstance(msg, Mapping):
        raise ValueError(f"orderbook message must be a mapping, got {msg!r}")
    msg_type = _msg_get(msg, "type")

    # Control frames carry no book data — ignore (W1).
    if msg_type in CONTROL_FRAME_TYPES:
        return

    if msg_type == _TYPE_SNAPSHOT:
        body = _body_of(msg)
        book.reset(
            int(_msg_get(msg, "seq")),  # the per-subscription seq anchor (D-02).
            yes=parse_dollar_fp_side(body.get("yes_dollars_fp")),
            no=parse_dollar_fp_side(body.get("no_dollars_fp")),
            ticker=_ticker_of(body),
        )
        # Carry a snapshot ts/ts_ms if present, else leave unchanged (never back-date, D-08).
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
        # Carry the WS event time onto the book (PAP-03); absent → leave unchanged.
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
    "CONTROL_FRAME_TYPES",
    "OrderBook",
    "SeqGap",
    "apply",
    "parse_dollar_fp_side",
]
