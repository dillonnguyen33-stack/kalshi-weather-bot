"""Engine + typed settings for the weatherquant ledger (D-09 / Pitfall 4 / ASVS V14).

`DATABASE_URL` is loaded from the environment (a local, git-ignored `.env` in dev) via
pydantic-settings and MUST use the ``postgresql+psycopg://`` dialect — psycopg v3, never
the legacy psycopg-2 driver (D-09). A field validator rejects any other scheme at
construction time (Pitfall 4). The URL is a secret: it is never logged or ``repr``'d
(ASVS V14) — the ``Settings`` repr is overridden so an accidental log line cannot leak
credentials.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine, create_engine

# The one required SQLAlchemy dialect — psycopg v3, never the legacy psycopg2 (D-09).
_REQUIRED_SCHEME = "postgresql+psycopg"

# The locked single-position cap band (PROJECT.md risk constraint / D-13): the configured
# max_position_fraction MUST land in [2%, 5%] of bankroll. The field validator rejects any
# value outside this inclusive band at construction so the hard cap can never be configured
# looser than policy (threat T-04-02).
_MIN_POSITION_FRACTION = 0.02
_MAX_POSITION_FRACTION = 0.05


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

    Phase-1 added ``database_url``; Phase-2 (02-01) adds the two ingestion secrets
    ``anthropic_api_key`` (AFD forecaster-disagreement classification, D-13) and
    ``wethr_api_key`` (Wethr.net bearer auth, ING-06). Both are nullable so ingestion
    degrades gracefully when a key is absent (D-11) — the AFD/Wethr clients skip rather
    than fail. The redacted ``__repr__``/``__str__`` below is a FIXED string, so adding
    secret fields here can never leak them in a log line (ASVS V14, threat T-02-01).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str
    anthropic_api_key: str | None = None
    wethr_api_key: str | None = None

    # Phase-4 money-path config (D-13). NOT secrets — these are policy numbers that may
    # appear in a normal repr; they are deliberately NOT added to the redacted ``__repr__``
    # below (that fixed string only hides credentials, ASVS V14).
    bankroll_usd: float = 500.0  # PROJECT.md constraint: the $500 paper-trading bankroll.
    max_position_fraction: float = 0.025  # D-13: conservative end of locked [0.02, 0.05].

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

    @field_validator("max_position_fraction")
    @classmethod
    def _validate_position_cap_band(cls, value: float) -> float:
        """Reject any single-position cap outside the locked ``[0.02, 0.05]`` band (D-13).

        Mirrors :meth:`_validate_psycopg3_scheme`'s fail-loud-at-construction shape. The cap
        is the hard money-path invariant (PROJECT.md risk constraint): an out-of-range value
        that silently raised position size is a money-path risk (threat T-04-02), so the cap
        can never be configured looser than policy. ``0.02`` and ``0.05`` are accepted.
        """
        if not (_MIN_POSITION_FRACTION <= value <= _MAX_POSITION_FRACTION):
            raise ValueError(
                f"max_position_fraction must be within the locked "
                f"[{_MIN_POSITION_FRACTION}, {_MAX_POSITION_FRACTION}] band (D-13). "
                f"Got: {value}"
            )
        return value

    def __repr__(self) -> str:  # never leak credential-bearing fields (ASVS V14)
        # Fixed string: no field VALUE is ever interpolated, so neither the URL nor the
        # API keys can leak through an accidental log/repr (threat T-02-01).
        return (
            "Settings(database_url=<redacted>, anthropic_api_key=<redacted>, "
            "wethr_api_key=<redacted>)"
        )

    __str__ = __repr__


def get_settings() -> Settings:
    """Construct ``Settings`` from the environment / ``.env`` (fails loud if unset)."""
    return Settings()  # type: ignore[call-arg]  # populated from env by pydantic-settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy ``Engine`` bound to ``DATABASE_URL``.

    Memoized: an ``Engine`` owns a connection pool, so every caller must share the one
    instance. Without this, a per-cycle caller (e.g. an APScheduler ingestion job in
    Phase 2+) would build a fresh pool — and re-read/re-validate ``.env`` — on each call.

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
