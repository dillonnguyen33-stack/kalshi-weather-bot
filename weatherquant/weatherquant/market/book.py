"""In-memory per-ticker orderbook + seq integrity (PAP-01, D-02).

The book is rebuilt from a Kalshi ``orderbook_snapshot`` (a full reset to a ``seq``), then
mutated by contiguous ``orderbook_delta`` messages. Each delta MUST carry ``book.seq + 1``;
any non-contiguous ``seq`` (a gap) raises :class:`SeqGap` — the book is then treated as
UNKNOWN, never carried forward silently (D-02, threat T-05-08). The client loop catches the
gap (it is a :class:`~weatherquant.ingest.errors.CorrectnessError`) and re-fetches the REST
snapshot as the resync anchor.

Message shape (RESEARCH Pattern 2 / Assumptions A1/A2 — MEDIUM confidence, pinned by the
05-02 live/demo checkpoint):

* ``orderbook_snapshot``: ``{"type", "seq", <ticker key>, "yes": [[price_cents, count], ...],
  "no": [[price_cents, count], ...]}`` — BID levels only on both sides (no asks; the ask is
  reflected as ``100 - opposite bid`` in :mod:`weatherquant.market.reflect`).
* ``orderbook_delta``: ``{"type", "seq", <ticker key>, "side", "price", "delta"}`` — a single
  signed size change at one ``(side, price_cents)`` level. ``+delta`` adds size, ``-delta``
  removes; a level whose total reaches ``0`` (or below) is dropped.

Defensive parsing (ASVS V5, threat T-05-10): the field names above are MEDIUM confidence, so
the ticker key is accepted as ``market_ticker`` OR ``market_id`` OR ``ticker``; an unknown
message ``type``, a missing required field, or a ``delta`` arriving before any ``snapshot``
FAILS LOUD — a malformed message is never coerced into a fabricated level (absence = absence).
The internal book is uniformly integer CENTS so the WS (cents) and REST (dollars→cents,
:mod:`weatherquant.market.client`) encodings converge to ONE representation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from weatherquant.ingest.errors import CorrectnessError

# The documented message types (RESEARCH Pattern 2).
_TYPE_SNAPSHOT = "orderbook_snapshot"
_TYPE_DELTA = "orderbook_delta"

# The two book sides (Kalshi quotes the BID side of each outcome).
_SIDES = ("yes", "no")

# The ticker key is MEDIUM-confidence (A1); tolerate every documented spelling.
_TICKER_KEYS = ("market_ticker", "market_id", "ticker")


class SeqGap(CorrectnessError, ValueError):
    """A non-contiguous ``seq`` was observed — the local book is UNKNOWN (D-02, T-05-08).

    Subclasses :class:`~weatherquant.ingest.errors.CorrectnessError` so the client's
    correctness-catch re-snapshots rather than silently applying an out-of-order delta, and
    :class:`ValueError` so a direct ``pytest.raises(ValueError)`` still holds (mirrors the
    ingest errors' dual base). Carries the ``(expected, got)`` seqs for diagnosis.
    """

    def __init__(self, expected: int, got: int) -> None:
        self.expected = expected
        self.got = got
        super().__init__(
            f"orderbook seq gap: expected {expected} (last seq + 1) but got {got} — "
            "the local book is unknown and must be re-snapshotted (D-02), never carried "
            "forward silently."
        )


def _coerce_levels(raw: Any) -> dict[int, int]:
    """Coerce a snapshot side (``[[price_cents, count], ...]``) to a ``{price: count}`` map.

    Fails loud (``ValueError``/``TypeError``) on a malformed level — a missing/non-numeric
    price or count is a real protocol breach, never silently coerced (ASVS V5, T-05-10).
    Non-positive counts are dropped (a level with no size is absence, not a fabricated level).
    """
    if raw is None:
        return {}
    if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes, Mapping)):
        raise ValueError(f"orderbook side must be a list of [price, count] levels, got {raw!r}")
    levels: dict[int, int] = {}
    for level in raw:
        if not isinstance(level, Iterable) or isinstance(level, (str, bytes)):
            raise ValueError(f"orderbook level must be [price, count], got {level!r}")
        items = list(level)
        if len(items) != 2:
            raise ValueError(f"orderbook level must be exactly [price, count], got {items!r}")
        price = int(items[0])
        count = int(items[1])
        if count > 0:
            levels[price] = count
    return levels


class OrderBook:
    """An in-memory per-ticker book: a ``seq`` baseline + a ``{price_cents: count}`` map per side.

    The two sides are stored internally as ``{price_cents: count}`` maps (``_yes`` / ``_no``)
    for O(1) delta application, but EXPOSED as the :attr:`yes` / :attr:`no` bid-level lists of
    ``(price_cents, count)`` pairs the reflection seam (:mod:`weatherquant.market.reflect`)
    iterates over — so ``yes_ask_levels(book)`` / ``no_ask_levels(book)`` work directly on this
    in-memory book exactly as they do on the dict fixtures (05-01 D: the dict-or-attribute
    accessor). Kalshi quotes only BIDS; the ask is reflected as ``100 - opposite bid``.

    A fresh ``OrderBook()`` is UNINITIALIZED (``seq is None``) — a delta arriving before any
    snapshot raises (fail loud, never coerce). :func:`apply` is the single mutation entry
    point; callers do not poke the maps directly.
    """

    __slots__ = ("ticker", "seq", "_yes", "_no")

    def __init__(self, ticker: str | None = None) -> None:
        self.ticker: str | None = ticker
        self.seq: int | None = None
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
        """Full reset to a snapshot: replace ``seq`` and both side maps (D-02)."""
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


def _ticker_of(msg: Mapping[str, Any]) -> str | None:
    """Return the ticker under whichever documented key is present (A1 — tolerate spellings)."""
    for key in _TICKER_KEYS:
        if key in msg:
            return msg[key]
    return None


def apply(book: OrderBook, msg: Mapping[str, Any]) -> None:
    """Apply one WS ``orderbook_snapshot`` / ``orderbook_delta`` message to ``book``.

    * ``orderbook_snapshot`` → full reset of ``(seq, yes, no)`` from the message.
    * ``orderbook_delta`` with ``seq == book.seq + 1`` → mutate the named ``(side, price)`` by
      the signed ``delta`` (drop a level at total 0) and advance ``book.seq``.
    * ``orderbook_delta`` with a NON-contiguous ``seq`` → raise :class:`SeqGap` (D-02).
    * A ``delta`` before any ``snapshot`` (uninitialized book) → raise (fail loud).
    * An unknown ``type`` / missing field → raise (ASVS V5, never coerce, T-05-10).

    The book is mutated in place.
    """
    if not isinstance(msg, Mapping):
        raise ValueError(f"orderbook message must be a mapping, got {msg!r}")
    msg_type = _msg_get(msg, "type")

    if msg_type == _TYPE_SNAPSHOT:
        book.reset(
            int(_msg_get(msg, "seq")),
            yes=msg.get("yes"),
            no=msg.get("no"),
            ticker=_ticker_of(msg),
        )
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
        book.apply_delta(
            side=_msg_get(msg, "side"),
            price=int(_msg_get(msg, "price")),
            delta=int(_msg_get(msg, "delta")),
        )
        book.seq = got_seq
        return

    raise ValueError(
        f"unknown orderbook message type {msg_type!r} — expected "
        f"{_TYPE_SNAPSHOT!r} or {_TYPE_DELTA!r} (fail loud, never coerce; T-05-10)."
    )


__all__ = ["OrderBook", "SeqGap", "apply"]
