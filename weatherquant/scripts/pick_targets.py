#!/usr/bin/env python3
"""Forecast-driven ticker picker — write today's paper-trading targets from the model's high.

For each registry city it computes the blended predicted daily high (reusing the money-path
``cli.pricing._blend_distribution`` — no re-derived math), lists today's OPEN Kalshi KXHIGH
buckets (public GET, no auth), and writes the ``--count`` buckets nearest the forecast to the
targets file that ``daily_paper.sh`` reads.

It does NOT compute EV: the ``paper --watch`` loop already gates fills on live EV>0 against the
real book, so pre-screening here would only duplicate (and desync) that decision. The picker's
job is narrower — surface the markets worth watching (the ones straddling the forecast); buckets
far from the predicted high have ~0 model probability and the watch loop won't fill them anyway.
# ponytail: proximity selection, not EV. Move EV here only if watching too many dead markets.

    uv run python scripts/pick_targets.py                 # today, 4 nearest, prod hosts
    uv run python scripts/pick_targets.py --date 2026-07-01 --count 3 --demo
    uv run python scripts/pick_targets.py --selftest      # pure-logic check, no network/DB
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
from typing import Any

logger = logging.getLogger("pick_targets")

_STRIKE_BETWEEN = {"between", "range", "in_range"}
_STRIKE_GREATER = {"greater", "greater_or_equal", "above"}
_STRIKE_LESS = {"less", "less_or_equal", "below"}


def kalshi_date_code(d: _dt.date) -> str:
    """Kalshi KXHIGH ticker date code, e.g. 2026-06-30 -> '26JUN30' (YYMMMDD, upper)."""
    return d.strftime("%y%b%d").upper()


def rep_temp(market: dict[str, Any]) -> float | None:
    """A representative °F for a market, for proximity ranking (mid of a range, else the strike)."""
    st = (market.get("strike_type") or "").lower()
    fs, cs = market.get("floor_strike"), market.get("cap_strike")
    if st in _STRIKE_BETWEEN and fs is not None and cs is not None:
        return (float(fs) + float(cs)) / 2.0
    if st in _STRIKE_GREATER and fs is not None:
        return float(fs)
    if st in _STRIKE_LESS and cs is not None:
        return float(cs)
    if fs is not None:
        return float(fs)
    if cs is not None:
        return float(cs)
    return None


def select_nearest(markets: list[dict[str, Any]], mu: float, k: int) -> list[str]:
    """The ``k`` market tickers whose representative °F is closest to ``mu`` (ticker tie-break)."""
    scored: list[tuple[float, str]] = []
    for m in markets:
        r = rep_temp(m)
        ticker = m.get("ticker")
        if r is None or not ticker:
            continue
        scored.append((abs(r - mu), ticker))
    scored.sort(key=lambda x: (x[0], x[1]))  # deterministic: distance, then ticker
    return [t for _, t in scored[:k]]


def _fetch_open_markets(rest_host: str, series: str) -> list[dict[str, Any]]:
    """GET the open markets for a KXHIGH series (public, unsigned). One page (limit 1000)."""
    import httpx

    url = f"{rest_host}/trade-api/v2/markets"
    params = {"series_ticker": series, "status": "open", "limit": 1000}
    resp = httpx.get(url, params=params, timeout=20.0)
    resp.raise_for_status()
    return resp.json().get("markets", []) or []


def pick(target: _dt.date, count: int, lead: int, demo: bool) -> list[tuple[str, str]]:
    """Return ``(city, ticker)`` picks: the ``count`` open buckets nearest each city's forecast."""
    from weatherquant.cli.pricing import _blend_distribution
    from weatherquant.db.engine import get_engine
    from weatherquant.market.client import _resolve_hosts
    from weatherquant.price.ticker import TICKER_CITY_SUFFIX_TO_KEY
    from weatherquant.registry import CITIES

    city_to_suffix = {v: k for k, v in TICKER_CITY_SUFFIX_TO_KEY.items()}
    _, rest_host = _resolve_hosts(demo)
    kdate = kalshi_date_code(target)
    bind = get_engine()

    picks: list[tuple[str, str]] = []
    for city in CITIES:
        suffix = city_to_suffix.get(city)
        if suffix is None:
            logger.warning("no Kalshi suffix for city=%s — skipping", city)
            continue
        try:
            blend = _blend_distribution(bind, city, target, lead)
        except SystemExit as exc:  # no forecasts/calibration for this city/date yet (D-11 shape)
            logger.warning("skip %s: no model distribution (%s)", city, exc)
            continue
        try:
            markets = _fetch_open_markets(rest_host, f"KXHIGH{suffix}")
        except Exception as exc:  # noqa: BLE001 — one city's market fetch failing must not abort the rest
            logger.warning("skip %s: market fetch failed (%s)", city, exc)
            continue
        today_markets = [m for m in markets if f"-{kdate}-" in (m.get("ticker") or "")]
        if not today_markets:
            logger.warning("skip %s: no open %s market for %s", city, f"KXHIGH{suffix}", kdate)
            continue
        chosen = select_nearest(today_markets, float(blend["mu_blend"]), count)
        logger.info(
            "%s mu=%.1f sigma=%.1f -> %d/%d buckets: %s",
            city, blend["mu_blend"], blend["sigma_blend"], len(chosen), len(today_markets), chosen,
        )
        picks.extend((city, t) for t in chosen)
    return picks


def write_targets(path: str, picks: list[tuple[str, str]], target: _dt.date, count: int) -> None:
    """Overwrite the targets file with the picked ``CITY TICKER`` lines (daily_paper.sh reads it)."""
    lines = [
        f"# Auto-generated by pick_targets.py for {target.isoformat()} "
        f"({count} nearest buckets/city). Re-run each trading day.",
    ]
    lines += [f"{city} {ticker}" for city, ticker in picks]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _selftest() -> None:
    """Pure-logic check (no network/DB): nearest-K ranking + date code + rep temp."""
    assert kalshi_date_code(_dt.date(2026, 6, 30)) == "26JUN30"
    mkts = [
        {"ticker": "B80", "strike_type": "between", "floor_strike": 79, "cap_strike": 81},
        {"ticker": "B83", "strike_type": "between", "floor_strike": 82, "cap_strike": 84},
        {"ticker": "B85", "strike_type": "between", "floor_strike": 84, "cap_strike": 86},
        {"ticker": "B87", "strike_type": "between", "floor_strike": 86, "cap_strike": 88},
        {"ticker": "T90", "strike_type": "greater", "floor_strike": 90, "cap_strike": None},
    ]
    assert rep_temp(mkts[2]) == 85.0
    assert rep_temp(mkts[4]) == 90.0
    # mu=85 -> distances: B85=0, B83=2, B87=2, B80=5, T90=5. Nearest 3 = B85 then B83/B87.
    got = select_nearest(mkts, 85.0, 3)
    assert got[0] == "B85", got
    assert set(got) == {"B85", "B83", "B87"}, got
    # A market with no usable strike is skipped, never crashes.
    assert select_nearest([{"ticker": "X", "strike_type": "between"}], 85.0, 3) == []
    print("pick_targets selftest: OK")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Write forecast-driven paper-trading targets.")
    p.add_argument("--date", default=None, help="Settlement date YYYY-MM-DD (default: today).")
    p.add_argument("--count", type=int, default=4, help="Buckets per city to watch (default 4).")
    p.add_argument(
        "--lead",
        type=int,
        default=24,
        help="Forecast lead hours — MUST match the lead you calibrated/trade at "
        "(this project fits + trades the 24h-ahead daily high, so default 24).",
    )
    p.add_argument("--out", default="scripts/paper_targets.txt", help="Targets file to write.")
    p.add_argument("--demo", action="store_true", help="Use Kalshi demo hosts.")
    p.add_argument("--selftest", action="store_true", help="Run the pure-logic check and exit.")
    args = p.parse_args(argv)

    if args.selftest:
        _selftest()
        return 0
    if args.count <= 0:
        raise SystemExit(f"pick_targets: --count must be positive, got {args.count}")

    target = (
        _dt.date.fromisoformat(args.date) if args.date else _dt.date.today()  # noqa: DTZ011
    )
    picks = pick(target, args.count, args.lead, args.demo)
    write_targets(args.out, picks, target, args.count)
    print(f"pick_targets: wrote {len(picks)} target(s) for {target.isoformat()} -> {args.out}")
    if not picks:
        print("  (no picks — ingest today's forecasts first, and check markets are open)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
