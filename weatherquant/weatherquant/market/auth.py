"""The ONE RSA-PSS request signer shared by the WS feed and REST snapshot (PAP-01, D-01).

Signs ``RSA-SHA256``/``PSS`` over ``timestamp_ms + method + path`` (path query-stripped,
Pitfall 6). Discipline (D-01; see docs/DECISIONS.md): key loaded by PATH only from
``Settings.kalshi_private_key_path`` (D-14), never stored/logged/repr'd (ASVS V14, T-05-05);
fail-loud on unset/missing path or malformed PEM; audited ``cryptography`` PSS primitive only,
never hand-rolled (T-05-06b).
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

    Errors reference the PATH only, never key material (ASVS V14, T-05-05).

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
        # PATH only, never key material (ASVS V14).
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
    """Return ``path`` with any ``?query`` suffix removed (Pitfall 6; signing with the query
    attached yields an invalid signature and a 401 / reconnect storm, T-05-06)."""
    return path.split("?", 1)[0]


def sign(private_key: RSAPrivateKey, key_id: str, method: str, path: str) -> dict[str, str]:
    """Sign ``(timestamp_ms + method + path)`` and return the three Kalshi auth headers (D-01).

    ``path`` is query-stripped before signing (Pitfall 6).

    Args:
        private_key: the RSA private key (from :func:`load_key`).
        key_id: the Kalshi access key id (``Settings.kalshi_key_id``).
        method: the HTTP method (e.g. ``"GET"``).
        path: the request path; any ``?query`` suffix is stripped before signing.

    Returns:
        A dict with exactly ``KALSHI-ACCESS-KEY`` / ``-SIGNATURE`` (base64) / ``-TIMESTAMP``
        (millisecond integer string).
    """
    ts = str(int(time.time() * 1000))  # epoch ms as a string
    signed_path = _strip_query(path)  # query-stripped (Pitfall 6)
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

    Holds the key OBJECT, never raw bytes; ``repr``/``str`` carries no key material
    (ASVS V14, T-05-05). Build via :meth:`from_settings` to load the key by path (fail-loud).
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
        """Build a signer from a ``Settings``-shaped object (key by path, fail-loud)."""
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
        # No key material or key id in repr (ASVS V14, T-05-05).
        return "KalshiSigner(key_id=<redacted>, private_key=<redacted>)"

    __str__ = __repr__


__all__ = ["KalshiSigner", "load_key", "sign"]
