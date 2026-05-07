"""Unit tests for the FAISS-based retrieval index."""

from pathlib import Path

import numpy as np
import pytest
from retrieval.config import RetrievalConfig

from retrieval.index import FAISS_AVAILABLE, CaseIndex


class DummyEmbeddingModel:
    def __init__(self, dimension: int = 4):
        self._dimension = dimension

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        vectors = []
        for text in texts:
            base = np.array(
                [len(text), text.count("contract"), text.count("plaintiff")
                 + text.count("defendant"), 1],
                dtype=np.float32,
            )
            norm = np.linalg.norm(base)
            vectors.append(base / (norm if norm > 0 else 1.0))
        return np.vstack(vectors)

    def embed(self, text: str) -> np.ndarray:
        vector = np.array(
            [len(text), text.count("contract")
             , text.count("plaintiff") + text.count("defendant"), 1],
            dtype=np.float32,
        )
        norm = np.linalg.norm(vector)
        return vector / (norm if norm > 0 else 1.0)

    @property
    def dimension(self) -> int:
        return self._dimension


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS support required")
class TestCaseIndex:
    def test_build_search_save_load(self, tmp_path: Path):
        config = RetrievalConfig(index_path=str(tmp_path / "retrieval"), similarity_threshold=0.0)
        embedding_model = DummyEmbeddingModel()
        index = CaseIndex(config=config, embedding_model=embedding_model)

        texts = [
            "The plaintiff sues for breach of contract.",
            "Defendant denies the contract existed and disputes liability.",
        ]
        case_numbers = ["CSM001", "CSM002"]
        case_titles = ["Contract dispute", "Liability defense"]

        index.build(texts=texts, case_numbers=case_numbers, case_titles=case_titles)
        assert index.size == 2

        results = index.search("Breach of contract and liability")
        assert len(results) >= 1
        assert results[0].case_number in case_numbers
        assert "Found" in index.explain(results)

        index.save()
        loaded = CaseIndex(config=config, embedding_model=embedding_model)
        loaded.load(tmp_path / "retrieval")
        assert loaded.size == 2
        loaded_results = loaded.search("Breach of contract and liability")
        assert len(loaded_results) >= 1
        assert loaded_results[0].case_number == results[0].case_number
