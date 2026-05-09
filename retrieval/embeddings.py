"""Embedding generation for case text using sentence-transformers."""

from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval.config import RetrievalConfig

logger = logging.getLogger(__name__)


import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, config):
        self.model = SentenceTransformer(config.embedding_model)

    # -------------------------
    # batch embeddings
    # -------------------------
    def embed_documents(self, texts: list[str]):
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    # -------------------------
    # single query embedding
    # -------------------------
    def embed_query(self, text: str):
        return self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]