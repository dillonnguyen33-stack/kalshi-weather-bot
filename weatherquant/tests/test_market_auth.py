"""GREEN — the ONE RSA-PSS request signer (PAP-01, D-01).

The Kalshi WS handshake / any custom REST call is signed manually with ``cryptography``
``PSS(MGF1(SHA256), DIGEST_LENGTH)`` over ``ts(ms) + method + path`` — the path WITHOUT
query params (Pitfall 6). These tests generate a throwaway RSA key in-process, point the
signer at it, and verify the produced signature against the matching public key over the
query-stripped message — no live Kalshi creds required. They also pin the secret-handling
discipline: fail loud on a missing/unset key path, and never leak key material via repr/str.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from weatherquant.market import auth

KEY_ID = "test-key-id-1234"

_PSS = padding.PSS(
    mgf=padding.MGF1(hashes.SHA256()),
    salt_length=padding.PSS.DIGEST_LENGTH,
)


@pytest.fixture
def rsa_keypair():
    """A throwaway 2048-bit RSA keypair generated in-process (no live creds)."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


@pytest.fixture
def key_file(tmp_path: Path, rsa_keypair) -> Path:
    """Write the throwaway private key to a PEM file the signer can load by path."""
    private_key, _ = rsa_keypair
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    path = tmp_path / "kalshi_private_key.pem"
    path.write_bytes(pem)
    return path


def test_signer_produces_verifiable_signature_over_ts_method_path(key_file, rsa_keypair):
    """The signature verifies under the public key over ``ts+GET+path`` (path sans query)."""
    _, public_key = rsa_keypair
    signer = auth.KalshiSigner(auth.load_key(str(key_file)), KEY_ID)

    method, path = "GET", "/trade-api/ws/v2"
    headers = signer.sign(method, path)

    # Exactly the three KALSHI-ACCESS-* headers, with the configured key id.
    assert set(headers) == {
        "KALSHI-ACCESS-KEY",
        "KALSHI-ACCESS-SIGNATURE",
        "KALSHI-ACCESS-TIMESTAMP",
    }
    assert headers["KALSHI-ACCESS-KEY"] == KEY_ID

    # -TIMESTAMP is a millisecond integer string close to now.
    ts = headers["KALSHI-ACCESS-TIMESTAMP"]
    assert ts.isdigit()
    ts_ms = int(ts)
    now_ms = int(time.time() * 1000)
    assert abs(now_ms - ts_ms) < 5000  # within a few seconds

    # -SIGNATURE is valid base64 and verifies over ts + method + path.
    signature = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])
    message = (ts + method + path).encode("utf-8")
    public_key.verify(signature, message, _PSS, hashes.SHA256())  # raises if invalid


def test_signer_path_excludes_query_string(key_file, rsa_keypair):
    """The signed message uses the path WITHOUT the query string (Pitfall 6)."""
    _, public_key = rsa_keypair
    signer = auth.KalshiSigner(auth.load_key(str(key_file)), KEY_ID)

    method = "GET"
    path_with_query = "/trade-api/v2/markets/T/orderbook?depth=10"
    stripped_path = "/trade-api/v2/markets/T/orderbook"

    headers = signer.sign(method, path_with_query)
    ts = headers["KALSHI-ACCESS-TIMESTAMP"]
    signature = base64.b64decode(headers["KALSHI-ACCESS-SIGNATURE"])

    # Verifies over the QUERY-STRIPPED message ...
    public_key.verify(
        signature, (ts + method + stripped_path).encode("utf-8"), _PSS, hashes.SHA256()
    )

    # ... and does NOT verify over the message WITH the query string attached.
    with pytest.raises(InvalidSignature):
        public_key.verify(
            signature,
            (ts + method + path_with_query).encode("utf-8"),
            _PSS,
            hashes.SHA256(),
        )


def test_timestamps_advance_between_calls(key_file):
    """Two calls a moment apart carry non-decreasing ms timestamps (real wall clock)."""
    signer = auth.KalshiSigner(auth.load_key(str(key_file)), KEY_ID)
    ts1 = int(signer.sign("GET", "/a")["KALSHI-ACCESS-TIMESTAMP"])
    time.sleep(0.002)
    ts2 = int(signer.sign("GET", "/a")["KALSHI-ACCESS-TIMESTAMP"])
    assert ts2 >= ts1


def test_missing_key_file_fails_loud(tmp_path: Path):
    """An unknown/missing private-key path raises (fail-loud, never a silent default)."""
    with pytest.raises(FileNotFoundError):
        auth.load_key(str(tmp_path / "does_not_exist.pem"))


def test_unset_key_path_fails_loud():
    """An unset (None/empty) key path raises rather than defaulting silently."""
    with pytest.raises(ValueError):
        auth.load_key(None)
    with pytest.raises(ValueError):
        auth.load_key("")


def test_malformed_pem_fails_loud(tmp_path: Path):
    """A malformed PEM raises rather than producing a half-built signer."""
    bad = tmp_path / "bad.pem"
    bad.write_bytes(b"not a real pem")
    with pytest.raises(ValueError):
        auth.load_key(str(bad))


def test_signer_repr_carries_no_key_material(key_file):
    """Neither the key id value nor any key material appears in repr/str (ASVS V14)."""
    signer = auth.KalshiSigner(auth.load_key(str(key_file)), KEY_ID)
    text = repr(signer) + str(signer)
    assert KEY_ID not in text
    assert "BEGIN" not in text  # no PEM material
    assert "<redacted>" in text


def test_from_settings_loads_by_path(key_file):
    """from_settings reads kalshi_key_id / kalshi_private_key_path and loads by path."""

    class _S:
        kalshi_key_id = KEY_ID
        kalshi_private_key_path = str(key_file)

    signer = auth.KalshiSigner.from_settings(_S())
    headers = signer.sign("GET", "/trade-api/ws/v2")
    assert headers["KALSHI-ACCESS-KEY"] == KEY_ID


def test_from_settings_fails_loud_when_unset():
    """from_settings raises when the key id or key path is unset (no silent default)."""

    class _Empty:
        kalshi_key_id = None
        kalshi_private_key_path = None

    with pytest.raises(ValueError):
        auth.KalshiSigner.from_settings(_Empty())
