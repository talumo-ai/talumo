"""Application settings loaded from environment variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LiteLLM gateway (all tiers route through LiteLLM)
    litellm_base_url: str = "http://localhost:4000/v1"
    litellm_local_model: str = "local"
    litellm_cheap_model: str = "cheap"
    litellm_frontier_model: str = "frontier"
    litellm_api_key: str = "sk-litellm"
    # Optional for local-only runs; required when remote tiers are used.
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")

    # Orchestrator
    max_tokens: int = 2048
    temperature: float = 0.2
    request_timeout: float = 120.0

    # Validation thresholds
    min_output_chars: int = 20
    max_output_chars: int = 50_000

    model_config = {"env_prefix": "TALUMO_"}


settings = Settings()
