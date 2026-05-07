"""Configuration for the retrieval module."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class RetrievalConfig(BaseSettings):
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL",
    )
    index_path: str = Field(default="data/retrieval_index", alias="RETRIEVAL_INDEX_PATH")
    top_k: int = 5
    similarity_threshold: float = 0.3
