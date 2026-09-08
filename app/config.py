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

    # OpenRouter routes a model across many providers whose throughput varies by
    # an order of magnitude. Measured on this model: default routing gave
    # 15.6 tok/s, throughput-sorted routing gave 130.3 tok/s. Empty disables the
    # preference and uses OpenRouter's default routing.
    openrouter_provider_sort: str = "throughput"

    # Reasoning effort: "low", "medium", "high", or "" to disable.
    #
    # Measured on a four-query benchmark (see PLAN.md "Reasoning calibration").
    # "low" and off both produced correct verdicts on all four; "high" produced
    # three of four, downgrading a genuinely compliant Mudarabah by inventing
    # CONDITIONAL findings about details the query never raised. Given more
    # budget the model manufactures doubt rather than reasoning more carefully,
    # which is corrosive here because the verdict rules already bias toward
    # NEEDS_REVIEW.
    #
    # "low" is the default because it matched off on verdicts while mapping
    # severities better (UNRESOLVED rather than CONDITIONAL for a clause the
    # corpus says to refer for review) and consolidating redundant findings.
    llm_reasoning_effort: str = "low"

    # Must comfortably exceed reasoning plus answer. Set too low, the reasoning
    # pass consumes the entire budget and the call returns empty content: 3072
    # of 3072 tokens were reasoning tokens before this was raised.
    llm_max_tokens: int = 12000

    # --- Qdrant ------------------------------------------------------------
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "aaoifi_ss_en"
    qdrant_timeout_seconds: float = 20.0

    # --- Retrieval ---------------------------------------------------------
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.35

    # --- Observability -----------------------------------------------------
    log_level: str = "INFO"

    # Log the fully resolved LLM prompt and the raw completion.
    #
    # On by default because an assessment must be reproducible from its trace:
    # without the exact prompt, a wrong verdict cannot be attributed to
    # retrieval, to the prompt, or to the model.
    #
    # Turn this OFF in production. A compliance query can carry client names,
    # deal terms, or material non-public information, and this writes it to
    # stdout and to the log file in plaintext. See DOCUMENTATION.md §5, Risk 3.
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
