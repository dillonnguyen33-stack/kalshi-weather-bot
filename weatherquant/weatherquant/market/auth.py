"""The ONE RSA-PSS request signer for Kalshi (PAP-01, D-01).

Kalshi authenticates every signed request with an ``RSA-SHA256`` signature using ``PSS``
padding over the message ``timestamp_ms + method + path`` (the path WITHOUT any query
string — Pitfall 6). The official REST SDK signs internally, but the WebSocket handshake
and any custom REST call we make are signed HERE — there is exactly one signer shared by
the WS feed and the REST snapshot, so the signing scheme is derived in exactly one place.

Secret-handling discipline (mirrors ``db/engine.py`` / ``ingest/available_at.py``):

* The RSA private key is loaded by PATH ONLY, from ``Settings.kalshi_private_key_path``
  (D-14) — never from key material in ``.env``/repo (loss is irrecoverable, leak is
  account-level trading access). The key object lives on the signer; the raw key bytes are
  never stored, returned, logged, or interpolated into ``repr``/``str`` (ASVS V14,
  threat T-05-05).
* Fail loud on unknown (mirrors ``available_at`` raising ``KeyError``): an unset or missing
  key path, or a malformed PEM, raises immediately — never a silent default. Exceptions
  reference the PATH, never the key material.
* The cryptographic primitive is the audited ``cryptography`` ``PSS(MGF1(SHA256),
  DIGEST_LENGTH)`` only — never a hand-rolled RSA/PSS implementation (threat T-05-06b,
  RESEARCH "Don't Hand-Roll").
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

# The three Kalshi auth header names produced by every signed request.
_HEADER_KEY = "KALSHI-ACCESS-KEY"
_HEADER_SIGNATURE = "KALSHI-ACCESS-SIGNATURE"
_HEADER_TIMESTAMP = "KALSHI-ACCESS-TIMESTAMP"


def load_key(path: str | None) -> RSAPrivateKey:
    """Load the RSA private key from a filesystem ``path`` (fail-loud-on-unknown, D-14).

    The path comes from ``Settings.kalshi_private_key_path`` — a file OUTSIDE the repo. An
    unset (``None``/empty) or missing path raises immediately, never a silent default
    (mirrors :func:`weatherquant.ingest.available_at.available_at` raising on an unknown
    model). A malformed/non-RSA PEM also raises. The error message references the PATH only,
    never the key material (ASVS V14, threat T-05-05).

    Args:
        path: filesystem path to the PEM-encoded RSA private key.

    Returns:
        The loaded ``RSAPrivateKey``.

    Raises:
        ValueError: if ``path`` is unset/empty.
        FileNotFoundError: if no file exists at ``path``.
        ValueError: (from ``cryptography``) if the PEM is malformed or not an RSA key.
    """
    if not path:
        raise ValueError(
            "kalshi_private_key_path is unset — the RSA private key path must be "
            "configured (Settings.kalshi_private_key_path); no silent default (D-14)."
        )
    key_path = Path(path)
    if not key_path.is_file():
        # Reference the PATH only — never the (absent) key material (ASVS V14).
        raise FileNotFoundError(
            f"kalshi private key not found at configured path: {key_path}"
        )
    key = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    if not isinstance(key, RSAPrivateKey):
        raise ValueError(
            f"key at {key_path} is not an RSA private key (Kalshi signing requires RSA-PSS)."
        )
    return key


def _strip_query(path: str) -> str:
    """Return ``path`` with any ``?query`` suffix removed (Pitfall 6).

    The signed message uses the path WITHOUT query params — signing
    ``/trade-api/v2/markets/T/orderbook?depth=10`` with the query attached produces an
    invalid signature and a 401 / reconnect storm (threat T-05-06).
    """
    return path.split("?", 1)[0]


def sign(private_key: RSAPrivateKey, key_id: str, method: str, path: str) -> dict[str, str]:
    """Sign ``(timestamp_ms + method + path)`` and return the three Kalshi auth headers.

    Shared by the WS handshake and the REST snapshot (D-01). The signed message strips any
    query string from ``path`` first (Pitfall 6). The signature uses the audited
    ``PSS(MGF1(SHA256), DIGEST_LENGTH)`` primitive (never hand-rolled, threat T-05-06b).

    Args:
        private_key: the RSA private key (from :func:`load_key`).
        key_id: the Kalshi access key id (``Settings.kalshi_key_id``).
        method: the HTTP method (e.g. ``"GET"``).
        path: the request path; any ``?query`` suffix is stripped before signing.

    Returns:
        A dict with exactly ``KALSHI-ACCESS-KEY`` / ``-SIGNATURE`` (base64) / ``-TIMESTAMP``
        (millisecond integer string).
    """
    ts = str(int(time.time() * 1000))  # milliseconds since epoch, as a string
    signed_path = _strip_query(path)  # path WITHOUT query params (Pitfall 6)
    message = (ts + method + signed_path).encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        _HEADER_KEY: key_id,
        _HEADER_SIGNATURE: base64.b64encode(signature).decode("ascii"),
        _HEADER_TIMESTAMP: ts,
    }


class KalshiSigner:
    """The ONE Kalshi request signer (WS handshake + REST snapshot share it, D-01).

    Construct from the loaded key + key id, or via :meth:`from_settings` which reads
    ``Settings.kalshi_key_id`` / ``Settings.kalshi_private_key_path`` and loads the key by
    path (fail-loud if unset/missing). The signer holds the key OBJECT — never the raw key
    bytes — and its ``repr``/``str`` carries no key material (ASVS V14, threat T-05-05).
    """

    __slots__ = ("_private_key", "_key_id")

    def __init__(self, private_key: RSAPrivateKey, key_id: str) -> None:
        if not key_id:
            raise ValueError(
                "kalshi_key_id is unset — the Kalshi access key id must be configured "
                "(Settings.kalshi_key_id); no silent default (D-14)."
            )
        self._private_key = private_key
        self._key_id = key_id

    @classmethod
    def from_settings(cls, settings: object) -> KalshiSigner:
        """Build a signer from a ``Settings``-shaped object (key by path, fail-loud).

        Reads ``kalshi_key_id`` and ``kalshi_private_key_path`` and loads the key via
        :func:`load_key`. Raises (never a silent default) when either is unset/missing.
        """
        key_id = getattr(settings, "kalshi_key_id", None)
        key_path = getattr(settings, "kalshi_private_key_path", None)
        if not key_id:
            raise ValueError(
                "kalshi_key_id is unset — configure Settings.kalshi_key_id (D-14)."
            )
        return cls(load_key(key_path), key_id)

    def sign(self, method: str, path: str) -> dict[str, str]:
        """Return the three Kalshi auth headers for ``(method, path)`` (query-stripped)."""
        return sign(self._private_key, self._key_id, method, path)

    def __repr__(self) -> str:
        # Never interpolate the key material or the key id value (ASVS V14, T-05-05).
        return "KalshiSigner(key_id=<redacted>, private_key=<redacted>)"

    __str__ = __repr__


__all__ = ["KalshiSigner", "load_key", "sign"]
