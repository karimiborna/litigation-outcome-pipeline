"""FAISS-based vector index for similar case retrieval."""

from __future__ import annotations

import json
import logging
from pathlib import Path

# Optional import for FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

import numpy as np

from retrieval.config import RetrievalConfig
from retrieval.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class CaseMetadataStore:
    """Stores case metadata alongside the FAISS index for retrieval results."""

    def __init__(self):
        self._cases: list[dict] = []

    def add(
        self, case_number: str, case_title: str, outcome: str | None = None, **extra: str
    ) -> int:
        idx = len(self._cases)
        self._cases.append(
            {
                "case_number": case_number,
                "case_title": case_title,
                "outcome": outcome,
                **extra,
            }
        )
        return idx

    def get(self, idx: int) -> dict:
        return self._cases[idx]

    def __len__(self) -> int:
        return len(self._cases)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self._cases, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> CaseMetadataStore:
        store = cls()
        store._cases = json.loads(path.read_text(encoding="utf-8"))
        return store


class SimilarCaseResult:
    """A single similar case retrieval result."""

    def __init__(self, case_number: str, case_title: str, score: float, metadata: dict):
        self.case_number = case_number
        self.case_title = case_title
        self.score = score
        self.metadata = metadata

    def to_dict(self) -> dict:
        return {
            "case_number": self.case_number,
            "case_title": self.case_title,
            "similarity_score": round(self.score, 4),
            **self.metadata,
        }


class CaseIndex:
    """FAISS vector index for finding similar historical cases."""

    def __init__(self, config: RetrievalConfig | None = None):
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available — case index will not work")
            self._config = config or RetrievalConfig()
            self._embedding_model = None
            self._index: None = None
            self._metadata = CaseMetadataStore()
            return
        
        self._config = config or RetrievalConfig()
        self._embedding_model = EmbeddingModel(config)
        self._index: faiss.IndexFlatIP | None = None
        self._metadata = CaseMetadataStore()

    def build(
        self,
        texts: list[str],
        case_numbers: list[str],
        case_titles: list[str],
        outcomes: list[str | None] | None = None,
    ) -> None:
        """Build the index from a list of case texts and metadata."""
        outcomes = outcomes or [None] * len(texts)
        logger.info("Building index from %d cases", len(texts))

        embeddings = self._embedding_model.embed_batch(texts)
        dimension = embeddings.shape[1]

        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(embeddings.astype(np.float32))

        self._metadata = CaseMetadataStore()
        for cn, ct, outcome in zip(case_numbers, case_titles, outcomes, strict=False):
            self._metadata.add(case_number=cn, case_title=ct, outcome=outcome)

        logger.info("Index built: %d vectors, dimension=%d", self._index.ntotal, dimension)

    def search(self, query_text: str, top_k: int | None = None) -> list[SimilarCaseResult]:
        """Find the top-K most similar cases to the query text."""
        if self._index is None or self._index.ntotal == 0:
            return []

        k = top_k or self._config.top_k
        k = min(k, self._index.ntotal)

        query_embedding = self._embedding_model.embed(query_text)
        query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

        scores, indices = self._index.search(query_embedding, k)

        results: list[SimilarCaseResult] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            if score < self._config.similarity_threshold:
                continue
            meta = self._metadata.get(int(idx))
            results.append(
                SimilarCaseResult(
                    case_number=meta["case_number"],
                    case_title=meta["case_title"],
                    score=float(score),
                    metadata={
                        k: v for k, v in meta.items() if k not in ("case_number", "case_title")
                    },
                )
            )

        return results

    def save(self, directory: str | Path | None = None) -> None:
        """Persist the FAISS index and metadata to disk."""
        if self._index is None:
            raise RuntimeError("No index to save — call build() first")

        path = Path(directory or self._config.index_path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path / "index.faiss"))
        self._metadata.save(path / "metadata.json")
        logger.info("Index saved to %s", path)

    def load(self, directory: str | Path | None = None) -> None:
        """Load a previously saved index from disk."""
        path = Path(directory or self._config.index_path)

        self._index = faiss.read_index(str(path / "index.faiss"))
        self._metadata = CaseMetadataStore.load(path / "metadata.json")
        logger.info("Index loaded: %d vectors", self._index.ntotal)

    @property
    def size(self) -> int:
        if self._index is None:
            return 0
        return self._index.ntotal
