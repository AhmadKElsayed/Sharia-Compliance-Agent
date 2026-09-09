"""Application configuration, loaded from the environment.

Every tunable lives here so that no module reaches into ``os.environ`` directly.
Settings are read once and cached; import ``get_settings()`` rather than the
module-level object so tests can override the cache.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Values come from the process environment, falling back to a local ``.env``
    file for development. Production (Render) sets them as dashboard secrets.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- OpenRouter --------------------------------------------------------
    openrouter_api_key: str = Field(default="", description="OpenRouter API key.")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "anthropic/claude-sonnet-4.5"
    openrouter_fallback_models: str = ""
    openrouter_embedding_model: str = "openai/text-embedding-3-large"
    embedding_dim: int = 3072
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2

    openrouter_provider_sort: str = "throughput"

    llm_reasoning_effort: str = "low"

    llm_max_tokens: int = 12000

    # --- Qdrant ------------------------------------------------------------
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "aaoifi_ss_en"
    qdrant_timeout_seconds: float = 20.0

    # --- Retrieval ---------------------------------------------------------
    retrieval_top_k: int = 10
    retrieval_score_threshold: float = 0.35

    rerank_enabled: bool = True
    rerank_model: str = "voyageai/rerank-2.5"
    rerank_candidates: int = 24

    # --- Observability -----------------------------------------------------
    log_level: str = "INFO"

    log_prompts: bool = True
    log_file: str = "logs/app.jsonl"

    @field_validator("llm_reasoning_effort")
    @classmethod
    def _validate_effort(cls, value: str) -> str:
        allowed = {"", "low", "medium", "high"}
        lowered = value.strip().lower()
        if lowered not in allowed:
            raise ValueError(
                f"llm_reasoning_effort must be one of {sorted(allowed)}, got {value!r}"
            )
        return lowered

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}")
        return upper

    @field_validator("qdrant_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @property
    def fallback_models(self) -> list[str]:
        """Fallback chat models, in the order they should be attempted."""
        return [m.strip() for m in self.openrouter_fallback_models.split(",") if m.strip()]

    @property
    def model_chain(self) -> list[str]:
        """Primary model followed by its fallbacks, de-duplicated."""
        chain: list[str] = []
        for model in [self.openrouter_model, *self.fallback_models]:
            if model and model not in chain:
                chain.append(model)
        return chain

    def missing_required(self) -> list[str]:
        """Names of required settings that are absent.

        Returned rather than raised so that ``/health`` can report configuration
        problems instead of the process failing to boot — a container that dies
        on a missing variable is harder to diagnose than one that reports why.
        """
        missing = []
        if not self.openrouter_api_key:
            missing.append("OPENROUTER_API_KEY")
        if not self.qdrant_url:
            missing.append("QDRANT_URL")
        if not self.qdrant_api_key:
            missing.append("QDRANT_API_KEY")
        return missing


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
