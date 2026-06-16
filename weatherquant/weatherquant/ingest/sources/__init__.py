"""Supplementary forecast sources (ING-04/05/06) behind ONE shared async httpx client.

Three Phase-2 supplementary blend inputs live here, each normalizing its provider-specific
JSON into Kelvin ``forecasts`` rows through 02-02's single audited write path
(``writer.insert_forecast`` + ``available_at`` + skip-before-insert idempotency):

* :mod:`weatherquant.ingest.sources.nws` — NWS gridpoint forecast bucketed into the LST
  ``settlement_window`` and stored under the provider-namespaced label ``nws`` (ING-04).
* :mod:`weatherquant.ingest.sources.openmeteo` — Open-Meteo ensemble members, each parsed
  in the API-default Celsius and converted to Kelvin (the v3 fahrenheit trap is NOT copied),
  stored per-member under ``openmeteo`` / ``openmeteo:<member>`` (ING-05).
* :mod:`weatherquant.ingest.sources.wethr` — Wethr.net deterministic outputs over bearer
  auth, provider-namespaced ``wethr:<model>``, optional key with graceful skip (ING-06).

THE SHARED CLIENT (D-14). v3 is sync ``requests`` per call; Phase 2 uses ONE async
``httpx.AsyncClient`` (HTTP/2, descriptive User-Agent, per-source timeouts) created by
:func:`weatherquant.ingest.sources._client.get_client`, plus a one-retry-on-429 backoff
helper. The v3 sync architecture is mined for endpoint constants ONLY — never reused.

PROVIDER NAMESPACING (D-12). The same underlying model from two providers (e.g. NOAA
``hrrr`` vs ``wethr:hrrr``) stays TWO distinct blend inputs — never deduped/merged.
"""

from __future__ import annotations
