"""Settings from environment variables (or a local .env that is never committed)."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore",
                                      protected_namespaces=("settings_",))

    # Model bundle: a version folder or a models/ root holding CURRENT. Set by the operator, never by a request.
    model_dir: Path = Path("models")
    app_env: Literal["production", "development"] = "production"

    # LLM explainer. Keys live on the server only.
    llm_provider: Literal["none", "openai", "anthropic"] = "none"
    llm_model: str = ""
    llm_base_url: str = ""
    openai_api_key: SecretStr = SecretStr("")
    anthropic_api_key: SecretStr = SecretStr("")
    llm_timeout_s: float = Field(20.0, gt=0, le=120)
    llm_max_tokens: int = Field(400, gt=0, le=2000)
    llm_max_retries: int = Field(2, ge=0, le=5)
    llm_stream: bool = True
    llm_cache_size: int = Field(256, ge=0)
    llm_session_limit_per_hour: int = Field(30, ge=1)
    prompts_dir: Path | None = None

    # HTTP hardening
    cors_origins: str = "http://localhost:5173"
    api_key: SecretStr = SecretStr("")
    rate_limit_per_minute: int = Field(120, ge=1)
    max_body_bytes: int = Field(65536, ge=1024)
    max_batch_rows: int = Field(1000, ge=1)
    explain_background_rows: int = Field(100, ge=10, le=1000)

    @field_validator("cors_origins")
    @classmethod
    def no_wildcard(cls, v: str) -> str:
        if "*" in v:
            raise ValueError("CORS_ORIGINS must list explicit origins; a wildcard is not allowed")
        return v

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_api_key(self) -> str:
        return {"openai": self.openai_api_key, "anthropic": self.anthropic_api_key}.get(self.llm_provider, SecretStr("")).get_secret_value()


@lru_cache
def get_settings() -> Settings:
    return Settings()
