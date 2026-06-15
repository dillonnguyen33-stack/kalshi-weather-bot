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

# The one required SQLAlchemy dialect — psycopg v3, never the legacy psycopg2 (D-09).
_REQUIRED_SCHEME = "postgresql+psycopg"


def require_psycopg3_scheme(url: str) -> None:
    """Raise ``ValueError`` unless ``url``'s dialect is exactly ``postgresql+psycopg``.

    Single source for the Pitfall-4 / D-09 guard, consumed by the ``Settings`` validator,
    the Alembic env, and the test conftest. The scheme is parsed from the part before
    ``://`` and compared by EXACT equality — a substring check like ``'+psycopg' in
    scheme`` wrongly accepts ``postgresql+psycopg2`` (the legacy driver), since
    ``'+psycopg'`` is a prefix of ``'+psycopg2'``. The error never echoes the credential
    (ASVS V14) — only the scheme is shown.
    """
    scheme = url.split("://", 1)[0] if "://" in url else url
    if scheme != _REQUIRED_SCHEME:
        raise ValueError(
            f"DATABASE_URL must use the '{_REQUIRED_SCHEME}://' dialect (psycopg v3 "
            f"only, never psycopg2). Got scheme: {scheme}://"
        )


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
    def _validate_psycopg3_scheme(cls, value: str) -> str:
        """Reject any DATABASE_URL whose dialect is not exactly ``postgresql+psycopg``.

        Delegates to the shared :func:`require_psycopg3_scheme` so the engine, the Alembic
        env, and the test conftest all enforce the identical exact-match check (Pitfall 4
        / D-09). A bare ``postgresql://`` or ``postgresql+psycopg2://`` URL is rejected.
        """
        require_psycopg3_scheme(value)
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

    ``execution_options(preserve_rowcount=True)`` is set on the engine so every INSERT
    reports a real ``result.rowcount`` (1 for a single row) despite the implicit
    ``RETURNING id`` on the ``Identity()`` PK (D-11 contract). Setting it on the app
    engine here — rather than as an import-time global listener on the Engine class —
    makes the behavior deterministic and not dependent on module import order.
    """
    settings = get_settings()
    return create_engine(
        settings.database_url,
        future=True,
        hide_parameters=True,
        execution_options={"preserve_rowcount": True},
    )
