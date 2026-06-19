"""End-to-end persist->CLV integration on PRODUCTION-shaped rows (PAP-04, 05-06 gap closure).

This is the test the isolated unit fixtures masked (the 05-VERIFICATION CR-01/WR-01 gap): it
exercises the FULL real money path on ONE consistent unit (cents ``mid`` + integer ``volume``)
through the actual audited writer and the actual read-back, NOT the hand-wired conftest
fixture:

    persist_snapshot (audited writer)            # write production-shaped rows
      -> market.persist.latest_snapshots         # read them BACK out of Postgres
        -> clv.closing_window_snapshots          # select the half-open LST closing window
          -> clv.vol_weighted_mid                # volume-weight the closing mid (CENTS)
            -> clv.clv_cents                      # subtract the fill's avg_price_cents (CENTS)

asserting a HAND-COMPUTED CLV. Because it reads the live ``volume`` column back, it would FAIL
against pre-fix code:
  * CR-01 (pre-fix ``mid`` persisted in [0,1] dollars) -> the closing mid is ~100x too small ->
    the hand-computed-CLV assertion fails.
  * WR-01 (pre-fix no ``volume`` column / never persisted) -> ``vol_weighted_mid`` raises
    ``KeyError('volume')`` on every read-back row.
It PASSES only with Tasks 1-2 applied AND the 0005 migration on the target DB.

DB-GATED: marked ``integration`` and routed through the ``pg_conn`` fixture, which depends on
``pg_engine`` — that fixture ``pytest.skip``s cleanly when ``DATABASE_URL`` is unset/unreachable
(see ``tests/conftest.py``). So the fast (no-DB) subset skips this cleanly rather than erroring;
it runs only once Postgres is reachable and migrated. Mirrors the integration-marker + ``pg_conn``
idiom in ``tests/test_writer_market.py``.

NOTE: ``pg_engine`` builds the schema from the Core metadata (``metadata.create_all``), so a
FRESH test DB already has ``market_snapshots.volume`` (models declares it). A PRE-EXISTING
operator DB needs the explicit ``0005`` migration (``uv run alembic upgrade head``) — that is the
operator-gated Task-3 step, not a test concern.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pytest

clv = pytest.importorskip("weatherquant.market.clv")
persist = pytest.importorskip("weatherquant.market.persist")

from weatherquant.registry import get_city  # noqa: E402
from weatherquant.time import settlement_window  # noqa: E402

pytestmark = pytest.mark.integration

# Reuse the NYC settlement date from tests/test_clv.py so the closing-window clock anchor is the
# same fixed-offset LST window the unit test pins (settlement_window(NYC, 2026-06-18).end_utc).
_CITY = "NYC"
_DAY = date(2026, 6, 18)
_TICKER = "KXHIGHNY-26JUN18-T72"


@dataclass(frozen=True)
class _Fill:
    """Minimal structural fill carrying the size-weighted avg price (cents) — matches clv."""

    avg_price_cents: float


def test_persist_to_clv_end_to_end_production_shaped(pg_conn):
    """persist -> latest_snapshots -> closing_window -> vol_weighted_mid -> clv_cents (cents+volume).

    Writes production-shaped snapshots (cents ``mid`` + integer ``volume``) through the REAL
    audited writer, reads them back out of Postgres via ``latest_snapshots`` (NOT the conftest
    fixture), selects the half-open closing window on the LST clock, and asserts a hand-computed
    CLV. FAILS against pre-fix code (CR-01 ~100x mid / WR-01 missing volume column); PASSES with
    the fix + applied migration.
    """
    win = settlement_window(get_city(_CITY), _DAY)
    end = win.end_utc
    window_minutes = clv.CLV_WINDOW_MINUTES

    # Two snapshots INSIDE the half-open closing window [end - CLV_WINDOW_MINUTES, end), plus one
    # BEFORE the window (must be excluded by closing_window_snapshots so it does not skew the mid).
    t_in_a = end - timedelta(minutes=window_minutes - 2)  # ~28m before end (inside)
    t_in_b = end - timedelta(minutes=window_minutes // 3)  # ~10m before end (inside)
    t_before = end - timedelta(minutes=window_minutes + 10)  # below the window (excluded)

    # Production-shaped rows: mid in FLOAT-VALUED CENTS (CR-01), volume an INTEGER count (WR-01).
    # Each needs a distinct snapshot_for (the natural key) so DISTINCT ON returns all three.
    rows = [
        # (available_at, mid_cents, volume) chosen so the in-window volume-weighted mid is exact:
        # (50*100 + 52*300) / (100+300) = (5000 + 15600) / 400 = 20600/400 = 51.5c.
        (t_in_a, 50.0, 100),
        (t_in_b, 52.0, 300),
        # An out-of-window row with a wildly different mid — must NOT enter the closing mid.
        (t_before, 10.0, 999),
    ]
    for available_at, mid_cents, volume in rows:
        rc = persist.persist_snapshot(
            pg_conn,
            ticker=_TICKER,
            snapshot_for=available_at.isoformat(),
            best_yes_bid=int(mid_cents) - 1,
            best_no_bid=100 - (int(mid_cents) + 1),
            mid=mid_cents,
            volume=volume,
            seq=None,
            detail={"raw": "book"},
            available_at=available_at,
        )
        assert rc == 1

    # Read the rows BACK out of Postgres through the real adapter (not the conftest fixture).
    persisted = persist.latest_snapshots(pg_conn, _TICKER)
    assert len(persisted) == 3  # one per distinct snapshot_for

    # Select the half-open closing window on the LST clock; the before-window row drops out.
    closing = clv.closing_window_snapshots(persisted, get_city(_CITY), _DAY)
    assert len(closing) == 2  # only the two in-window snapshots

    # Hand-computed volume-weighted closing mid over the read-back rows = 51.5c (CENTS).
    expected_closing_mid = (50.0 * 100 + 52.0 * 300) / (100 + 300)
    assert expected_closing_mid == pytest.approx(51.5)
    assert clv.vol_weighted_mid(closing) == pytest.approx(51.5)

    # A yes BUY filled at 48c is BETTER than the 51.5c close -> POSITIVE CLV = 51.5 - 48.0 = 3.5c.
    fill = _Fill(avg_price_cents=48.0)
    cl = clv.clv_cents(fill, closing, "buy")
    assert cl == pytest.approx(51.5 - 48.0)
    assert cl == pytest.approx(3.5)
    assert cl > 0.0  # would be ~ -48 (CR-01 ~100x mid) or raise KeyError (WR-01) pre-fix
