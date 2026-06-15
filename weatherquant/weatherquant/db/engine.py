"""Engine + typed settings for the weatherquant ledger (D-09 / Pitfall 4 / ASVS V14).

`DATABASE_URL` is loaded from the environment (a local, git-ignored `.env` in dev) via
pydantic-settings and MUST use the ``postgresql+psycopg://`` dialect — psycopg v3, never
the legacy psycopg-2 driver (D-09). A field validator rejects any other scheme at
construction time (Pitfall 4). The URL is a secret: it is never logged or ``repr``'d
(ASVS V14) — the ``Settings`` repr is overridden so an accidental log line cannot leak
credentials.
"""

from __future__ import annotations

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine, create_engine


class Settings(BaseSettings):
    """Typed application settings sourced from the environment / local ``.env``.

    Only the Phase-1 persistence config lives here; later phases add their own fields.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str

    @field_validator("database_url")
    @classmethod
    def _require_psycopg3_scheme(cls, value: str) -> str:
        """Reject any DATABASE_URL whose scheme is not ``postgresql+psycopg://``.

        Guards against the legacy-driver dialect leak (Pitfall 4 / D-09): a bare
        ``postgresql://`` URL defaults SQLAlchemy to the old psycopg-2 driver, which is
        forbidden and may not be installed. The error never echoes the credential.
        """
        scheme = value.split("://", 1)[0] if "://" in value else value
        if "+psycopg" not in scheme:
            raise ValueError(
                "DATABASE_URL must use the 'postgresql+psycopg://' dialect "
                f"(psycopg v3 only). Got scheme: {scheme}://"
            )
        return value

    def __repr__(self) -> str:  # never leak the credential-bearing URL (ASVS V14)
        return "Settings(database_url=<redacted>)"

    __str__ = __repr__


def get_settings() -> Settings:
    """Construct ``Settings`` from the environment / ``.env`` (fails loud if unset)."""
    return Settings()  # type: ignore[call-arg]  # populated from env by pydantic-settings


def get_engine() -> Engine:
    """Return a SQLAlchemy ``Engine`` bound to the validated ``DATABASE_URL``.

    The engine is built on the ``postgresql+psycopg://`` dialect. ``hide_parameters``
    keeps bound values out of error logs; the connection URL itself is never logged.
    """
    settings = get_settings()
    return create_engine(settings.database_url, future=True, hide_parameters=True)
