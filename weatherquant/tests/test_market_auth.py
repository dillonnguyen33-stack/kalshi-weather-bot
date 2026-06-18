"""RED stub — RSA-PSS request signer (PAP-01, 05-02 fills this GREEN).

The Kalshi WS handshake / any custom REST call is signed manually with ``cryptography``
``PSS(MGF1(SHA256), DIGEST_LENGTH)`` over ``ts(ms) + method + path`` — the path WITHOUT
query params (Pitfall 6). The official REST SDK signs internally, so there is no in-repo
analog; the signing body is RESEARCH §Code-Examples "RSA-PSS signer".

This is a Wave-0 RED stub: ``importorskip`` the not-yet-existing ``weatherquant.market.auth``
so collection succeeds and the fast subset stays green; 05-02 implements the module and the
test verifies a real signature against the public key.
"""

from __future__ import annotations

import pytest

auth = pytest.importorskip("weatherquant.market.auth")


@pytest.mark.xfail(reason="RED — 05-02 implements the RSA-PSS signer", strict=False)
def test_signer_produces_verifiable_signature_over_ts_method_path():
    """The signature verifies under the public key over ``ts+GET+path`` (path sans query)."""
    raise NotImplementedError("05-02: sign ts(ms)+method+path, verify with the public key")


@pytest.mark.xfail(reason="RED — 05-02 implements the RSA-PSS signer", strict=False)
def test_signer_path_excludes_query_string():
    """The signed message uses the path WITHOUT the query string (Pitfall 6)."""
    raise NotImplementedError("05-02: assert '?...' is stripped before signing")


@pytest.mark.xfail(reason="RED — 05-02 implements the RSA-PSS signer", strict=False)
def test_missing_key_file_fails_loud():
    """An unknown/missing private-key path raises (fail-loud-on-unknown, never a silent default)."""
    raise NotImplementedError("05-02: load from Settings.kalshi_private_key_path, raise if absent")
