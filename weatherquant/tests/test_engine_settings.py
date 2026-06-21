"""get_settings memoization — one .env parse per process, not per call (Phase-4 fix #1)."""

from __future__ import annotations


def test_get_settings_is_memoized(monkeypatch):
    """get_settings() returns the SAME Settings instance — parsed once, not re-read per call."""
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost/db")
    from weatherquant.db import engine

    engine.get_settings.cache_clear()  # only exists once memoized (RED before the fix)
    try:
        first = engine.get_settings()
        second = engine.get_settings()
        assert first is second
    finally:
        engine.get_settings.cache_clear()
