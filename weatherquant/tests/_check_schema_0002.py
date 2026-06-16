"""Plain assertion script (NOT a pytest file) verifying the 02-01 schema extensions.

Run with: ``uv run python tests/_check_schema_0002.py``

Asserts — purely from the in-memory SQLAlchemy Core metadata, no database needed —
that ``weatherquant.db.models`` carries the Phase-2 ingestion columns and that
``ix_forecasts_latest`` puts ``member`` between ``lead`` and ``available_at`` (D-05/06/07).
The migration 0002 column set must equal this metadata (create_all == migrated schema).
"""

from __future__ import annotations

import sys

from weatherquant.db.models import forecasts, observations


def _check() -> None:
    forecast_cols = set(forecasts.c.keys())
    expected_forecast = {
        "member",
        "temp_kelvin",
        "cycle",
        "station_lat",
        "station_lon",
        "grid_distance_m",
    }
    missing_f = expected_forecast - forecast_cols
    assert not missing_f, f"forecasts missing columns: {sorted(missing_f)}"

    obs_cols = set(observations.c.keys())
    expected_obs = {
        "daily_high_f",
        "window_start",
        "window_end",
        "obs_count",
        "detail",
    }
    missing_o = expected_obs - obs_cols
    assert not missing_o, f"observations missing columns: {sorted(missing_o)}"

    # ix_forecasts_latest column order must be exactly:
    # city, target_date, model, lead, member, available_at
    index = next(
        ix for ix in forecasts.indexes if ix.name == "ix_forecasts_latest"
    )
    order = [c.name for c in index.columns]
    expected_order = [
        "city",
        "target_date",
        "model",
        "lead",
        "member",
        "available_at",
    ]
    assert order == expected_order, (
        f"ix_forecasts_latest order is {order}, expected {expected_order}"
    )

    # member must be NOT NULL with a server default (D-05).
    member_col = forecasts.c["member"]
    assert member_col.nullable is False, "forecasts.member must be NOT NULL"
    assert member_col.server_default is not None, (
        "forecasts.member must have a server default (0)"
    )

    print("schema check OK: forecasts/observations extended, ix_forecasts_latest carries member")


if __name__ == "__main__":
    try:
        _check()
    except AssertionError as exc:
        print(f"SCHEMA CHECK FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
