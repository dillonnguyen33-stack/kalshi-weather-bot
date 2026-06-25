"""Engine + typed settings for the weatherquant ledger (D-09 / Pitfall 4 / ASVS V14).

`DATABASE_URL` (from env / git-ignored `.env`) MUST use the ``postgresql+psycopg://``
dialect (D-09); a validator rejects any other scheme. The URL is a secret, never
logged or ``repr``'d (ASVS V14).
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine, create_engine

# The one required SQLAlchemy dialect — psycopg v3, never the legacy psycopg2 (D-09).
_REQUIRED_SCHEME = "postgresql+psycopg"

# Locked single-position cap band [2%, 5%] of bankroll (D-13; threat T-04-02).
_MIN_POSITION_FRACTION = 0.02
_MAX_POSITION_FRACTION = 0.05

# Single source for the default single-position cap (D-13; WR-A1) — shared by the
# Settings field default and price.kelly.stake_fraction's cap default so they can't drift.
DEFAULT_POSITION_FRACTION = 0.025

# Locked set of allowed execution modes (D-15; threat T-05-02). Gates whether any live
# order path is reachable; the structural no-order-path guard lands in 05-03.
_ALLOWED_EXECUTION_MODES = frozenset({"paper", "live"})

# Default execution mode (D-15): paper-only this milestone (Gate 1).
DEFAULT_EXECUTION_MODE = "paper"

# Phase-6 drift monitor default (D-10): a trailing-window per-(city, model) reliability error (ECE)
# above this is a calibration-drift breach. 0.10 = 10 calibration points, a principled "the 70%
# bucket is now resolving ~60% or ~80%" alarm line; operator-tunable, NOT a secret (stays out of
# the redacted repr). Must lie in (0, 1] (a non-positive or >1 threshold is meaningless for ECE).
DRIFT_RELIABILITY_THRESHOLD_DEFAULT = 0.10


def require_psycopg3_scheme(url: str) -> None:
    """Raise ``ValueError`` unless ``url``'s dialect is exactly ``postgresql+psycopg``.

    Single source for the Pitfall-4 / D-09 guard (Settings validator, Alembic env, test
    conftest). EXACT-equality match — a substring check would wrongly accept the legacy
    ``postgresql+psycopg2``. The error shows only the scheme, never the credential (ASVS V14).
    """
    scheme = url.split("://", 1)[0] if "://" in url else url
    if scheme != _REQUIRED_SCHEME:
        raise ValueError(
            f"DATABASE_URL must use the '{_REQUIRED_SCHEME}://' dialect (psycopg v3 "
            f"only, never psycopg2). Got scheme: {scheme}://"
        )


class Settings(BaseSettings):
    """Typed application settings from env / local ``.env``.

    Secret fields are nullable so ingestion degrades gracefully when a key is absent
    (D-11). The redacted ``__repr__``/``__str__`` below is a FIXED string, so secret
    fields can never leak in a log line (ASVS V14, threat T-02-01).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str
    anthropic_api_key: str | None = None
    wethr_api_key: str | None = None

    # Phase-4 money-path config (D-13). NOT secrets — policy numbers, kept out of the repr.
    bankroll_usd: float = 500.0  # PROJECT.md constraint: the $500 paper-trading bankroll.
    max_position_fraction: float = DEFAULT_POSITION_FRACTION  # D-13: see DEFAULT_POSITION_FRACTION.

    # Phase-5 Kalshi credentials (D-14). SECRETS, nullable (supplied out-of-band before the
    # live checkpoints). ``kalshi_private_key_path`` is a PATH to the RSA key OUTSIDE the repo,
    # never key material in repo. Both added to the redacted ``__repr__`` (ASVS V14, T-05-01).
    kalshi_key_id: str | None = None
    kalshi_private_key_path: str | None = None
    # D-15: execution policy. NOT a secret (out of the repr) but validated to {paper, live}
    # below so it can't silently unlock a live path. Defaults to paper (Gate 1).
    execution_mode: str = DEFAULT_EXECUTION_MODE

    # Phase-6 drift monitor (D-10). NOT a secret — an operator-tunable policy number kept out of
    # the redacted repr. Trailing-window ECE above this per (city, model) is a drift breach
    # (SYS-02); validated to (0, 1] below.
    drift_reliability_threshold: float = DRIFT_RELIABILITY_THRESHOLD_DEFAULT

    # Phase-6 Gate-1 disjointness knob (D-10/D-12 anti-p-hacking, CR-04). The Phase-3 OOS
    # (calibration-tuning) slice as a half-open ``[start, end)`` date pair. NOT secrets — policy
    # dates kept out of the redacted repr. Nullable so an UNSET knob is detectable: run_verify is
    # FAIL-CLOSED — it raises rather than scoring the Gate-1 verdict on a None/empty tuning window
    # (a verdict scored on tuning data is a false "proof is valid", the worst possible error).
    verify_phase3_oos_start: date | None = None
    verify_phase3_oos_end: date | None = None

    @field_validator("database_url")
    @classmethod
    def _validate_psycopg3_scheme(cls, value: str) -> str:
        """Reject any DATABASE_URL whose dialect is not exactly ``postgresql+psycopg`` (D-09).

        Delegates to the shared :func:`require_psycopg3_scheme` (Pitfall 4).
        """
        require_psycopg3_scheme(value)
        return value

    @field_validator("max_position_fraction")
    @classmethod
    def _validate_position_cap_band(cls, value: float) -> float:
        """Reject any single-position cap outside the locked ``[0.02, 0.05]`` band (D-13).

        Fail-loud at construction: the cap can never be configured looser than policy
        (threat T-04-02). ``0.02`` and ``0.05`` are accepted.
        """
        if not (_MIN_POSITION_FRACTION <= value <= _MAX_POSITION_FRACTION):
            raise ValueError(
                f"max_position_fraction must be within the locked "
                f"[{_MIN_POSITION_FRACTION}, {_MAX_POSITION_FRACTION}] band (D-13). "
                f"Got: {value}"
            )
        return value

    @field_validator("execution_mode")
    @classmethod
    def _validate_execution_mode(cls, value: str) -> str:
        """Reject any ``execution_mode`` not in the locked ``{paper, live}`` set (D-15).

        Fail-loud at construction: an out-of-policy value must never silently unlock a live
        path (threat T-05-02). ``"paper"``/``"live"`` pass; the default is ``"paper"``.
        """
        if value not in _ALLOWED_EXECUTION_MODES:
            allowed = ", ".join(sorted(_ALLOWED_EXECUTION_MODES))
            raise ValueError(
                f"execution_mode must be one of {{{allowed}}} (D-15). Got: {value!r}"
            )
        return value

    @field_validator("drift_reliability_threshold")
    @classmethod
    def _validate_drift_threshold(cls, value: float) -> float:
        """Reject a drift-reliability threshold outside ``(0, 1]`` (D-10).

        Fail-loud at construction: ECE lives in ``[0, 1]``, so a non-positive threshold would alarm
        always and a >1 threshold could never alarm — both are operator misconfigurations (SYS-02).
        """
        if not (0.0 < value <= 1.0):
            raise ValueError(
                f"drift_reliability_threshold must be in (0, 1] (D-10). Got: {value}"
            )
        return value

    def __repr__(self) -> str:  # never leak credential-bearing fields (ASVS V14)
        # Fixed string: no field VALUE is interpolated, so no secret can leak via repr
        # (threats T-02-01, T-05-01). Policy fields (execution_mode etc.) stay OUT too.
        return (
            "Settings(database_url=<redacted>, anthropic_api_key=<redacted>, "
            "wethr_api_key=<redacted>, kalshi_key_id=<redacted>, "
            "kalshi_private_key_path=<redacted>)"
        )

    __str__ = __repr__


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Construct ``Settings`` from the environment / ``.env`` once per process (fails loud if unset).

    Memoized (like :func:`get_engine`) so callers share one parsed config and the ``.env`` is read
    once, not on every key lookup. ``cache_clear()`` is available for tests that repoint the env.
    """
    return Settings()  # type: ignore[call-arg]  # populated from env by pydantic-settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide SQLAlchemy ``Engine`` bound to ``DATABASE_URL``.

    Memoized so every caller shares the one connection pool (and one ``.env`` read).
    ``hide_parameters`` keeps bound values out of error logs. ``preserve_rowcount``
    makes every INSERT report a real ``result.rowcount`` despite the implicit
    ``RETURNING id`` on the ``Identity()`` PK — set here deterministically, not as an
    import-order-dependent global listener (D-11 contract).
    """
    settings = get_settings()
    return create_engine(
        settings.database_url,
        future=True,
        hide_parameters=True,
        execution_options={"preserve_rowcount": True},
    )
