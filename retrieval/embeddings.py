"""Embedding generation for case text using sentence-transformers."""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.config import RetrievalConfig

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wraps a sentence-transformers model for case text embedding."""

    def __init__(self, config: RetrievalConfig | None = None):
        self._config = config or RetrievalConfig()
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self._config.embedding_model)
            self._model = SentenceTransformer(self._config.embedding_model)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text string."""
        model = self._load_model()
        return model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a batch of texts."""
        model = self._load_model()
        return model.encode(texts, batch_size=batch_size, normalize_embeddings=True)

    @property
    def dimension(self) -> int:
        model = self._load_model()
        return model.get_sentence_embedding_dimension()
