"""The ONE shared async httpx client for every supplementary source (D-14).

v3 issued a fresh synchronous ``requests.get`` per source per call; Phase 2 routes NWS,
Open-Meteo, and Wethr through a SINGLE ``httpx.AsyncClient`` (HTTP/2, sane per-source
timeouts, a descriptive default User-Agent) created here. Centralizing the client means:

* the api.weather.gov User-Agent requirement (Pitfall 7 — NWS 403s without one) is set
  once as a default header rather than re-specified at every call site, and
* the 429 backoff policy (Pitfall 7 — v3 did one ``sleep(10)`` retry) lives in exactly one
  helper (:func:`request_with_retry`) instead of being copy-pasted per source.

The client is created per-call by default (``get_client()``), but every public source
function also accepts an injected ``client=`` so the unit tests can pass an
``httpx.AsyncClient(transport=httpx.MockTransport(...))`` and exercise the real request /
parse path with NO network (the established 02-03 obs/afd mocking idiom).

SSRF guard (T-02-15): this module never builds a base URL from untrusted input — each
source module owns its fixed endpoint constant; the client only carries transport policy.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

logger = logging.getLogger(__name__)

# Absolute-zero offset shared by every source's unit seam (°C/°F → Kelvin, D-07). One constant
# so the Kelvin-only forecast path has a single audited value, not four copies.
KELVIN_OFFSET = 273.15

# Descriptive default User-Agent. api.weather.gov REQUIRES a User-Agent (Pitfall 7) and it
# is courteous on the other feeds; carrying it as a client default means no source can
# forget it. (A contact is recommended by NWS; kept generic here, no secret.)
USER_AGENT = "weatherquant/0.1 (kalshi daily-high paper-trading)"

# Per-source default timeout. REST forecast payloads are small; 15s is generous while still
# failing rather than hanging the event loop.
DEFAULT_TIMEOUT = 15.0

# One backoff retry on HTTP 429 (Pitfall 7). v3 slept 10s; a shorter default keeps the async
# path responsive while still honoring the rate-limit signal.
_RETRY_BACKOFF_SECONDS = 10.0


def get_client(
    *,
    timeout: float = DEFAULT_TIMEOUT,
    headers: dict[str, str] | None = None,
) -> httpx.AsyncClient:
    """Return the shared async httpx client (HTTP/2, default User-Agent, per-source timeout).

    Args:
        timeout: per-request timeout in seconds (a source may pass a tighter/looser bound).
        headers: extra default headers merged over the User-Agent default (e.g. a Wethr
            ``Authorization: Bearer`` header). The User-Agent default is always present.

    Returns:
        An ``httpx.AsyncClient``. The caller owns the lifecycle (``async with`` / ``aclose``).
        HTTP/2 is requested but falls back to HTTP/1.1 automatically if the optional ``h2``
        extra is not installed, so this never hard-fails on a minimal install.
    """
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    try:
        return httpx.AsyncClient(timeout=timeout, headers=merged, http2=True)
    except ImportError:
        # The HTTP/2 extra (h2) is optional; degrade to HTTP/1.1 rather than fail (D-11).
        logger.debug("httpx h2 extra unavailable; using HTTP/1.1 for the shared client")
        return httpx.AsyncClient(timeout=timeout, headers=merged)


@asynccontextmanager
async def managed_client(
    client: httpx.AsyncClient | None = None,
) -> AsyncIterator[httpx.AsyncClient]:
    """Yield a client, closing it ONLY if we created it (the injected-client lifecycle, D-14).

    The single home for the ``owns_client = client is None; ... finally: if owns: aclose()``
    dance every source repeated: an injected test client is left open for its owner, a
    self-created one is closed on exit.
    """
    owns_client = client is None
    client = client or get_client()
    try:
        yield client
    finally:
        if owns_client:
            await client.aclose()


async def request_with_retry(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    backoff_seconds: float = _RETRY_BACKOFF_SECONDS,
    **kwargs: object,
) -> httpx.Response:
    """Issue one request, retrying EXACTLY ONCE after a backoff on HTTP 429 (Pitfall 7).

    A 429 (rate limited) triggers a single ``asyncio.sleep(backoff_seconds)`` then one retry
    — matching v3's one-shot ``sleep(10)`` retry but on the async loop. Any other status is
    returned as-is for the caller to ``raise_for_status``; non-429 errors are NOT retried
    here (a source decides its own degrade path, D-11). Returns the final response.
    """
    response = await client.request(method, url, **kwargs)  # type: ignore[arg-type]
    if response.status_code == 429:
        logger.warning(
            "HTTP 429 from %s — backing off %.1fs then retrying once (Pitfall 7)",
            url,
            backoff_seconds,
        )
        await asyncio.sleep(backoff_seconds)
        response = await client.request(method, url, **kwargs)  # type: ignore[arg-type]
    return response


__all__ = [
    "DEFAULT_TIMEOUT",
    "KELVIN_OFFSET",
    "USER_AGENT",
    "get_client",
    "managed_client",
    "request_with_retry",
]
