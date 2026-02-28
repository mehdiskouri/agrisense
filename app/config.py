"""Application settings loaded from environment variables via pydantic-settings."""

from enum import StrEnum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class FarmType(StrEnum):
    open_field = "open_field"
    greenhouse = "greenhouse"
    hybrid = "hybrid"


class LogFormat(StrEnum):
    json = "json"
    console = "console"


class Settings(BaseSettings):
    """Central configuration — all values sourced from env vars or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Database ────────────────────────────────────────────────────────────
    database_url: str = (
        "postgresql+asyncpg://agrisense:agrisense@localhost:5432/agrisense"
    )

    # ── Redis ───────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"

    # ── Auth ────────────────────────────────────────────────────────────────
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30

    # ── LLM ─────────────────────────────────────────────────────────────────
    anthropic_api_key: str = ""

    # ── Farm defaults ───────────────────────────────────────────────────────
    farm_default_type: FarmType = FarmType.greenhouse

    # ── Observability ───────────────────────────────────────────────────────
    log_level: str = "info"
    log_format: LogFormat = LogFormat.json

    # ── Julia ───────────────────────────────────────────────────────────────
    julia_project: str = "core/AgriSenseCore"
    julia_num_threads: str = "auto"


@lru_cache
def get_settings() -> Settings:
    """Singleton settings instance (cached after first call)."""
    return Settings()
