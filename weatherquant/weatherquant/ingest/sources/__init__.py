"""Supplementary forecast sources (ING-04/05/06) behind ONE shared async httpx client.

Three blend inputs — nws, openmeteo (Celsius→Kelvin, not the v3 °F trap), wethr (optional key
with graceful skip) — each normalizing provider JSON into Kelvin ``forecasts`` rows through the
single audited writer. All share one ``httpx.AsyncClient`` with a 429 backoff helper (D-14),
and provider namespacing keeps the same model from two providers as two distinct inputs (e.g.
``hrrr`` vs ``wethr:hrrr``, D-12; see docs/DECISIONS.md).
"""

from __future__ import annotations
