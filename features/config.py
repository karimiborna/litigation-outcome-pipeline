"""Configuration for the LLM-based feature extraction pipeline."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class FeaturesConfig(BaseSettings):
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_base_url: str = Field(default="", alias="LLM_BASE_URL")

    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_timeout: float = 60.0

    cache_dir: str = Field(default="data/features_cache", alias="FEATURES_CACHE_DIR")
    enable_cache: bool = True

    feature_version: str = "v1"
