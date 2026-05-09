"""
Production-ready Hybrid RAG Case Index

Components:
- Dense retrieval (FAISS)
- Sparse retrieval (BM25)
- Reciprocal Rank Fusion (RRF)
- Snapshot-based persistence (manifest + documents + FAISS)

Key design:
- Single embedding abstraction: EmbeddingModel
- Single source of truth: DocumentStore
- Fully reproducible load/save
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Any

import numpy as np
import faiss
from rank_bm25 import BM25Okapi

from retrieval.config import RetrievalConfig
from retrieval.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


# =========================================================
# MANIFEST
# =========================================================

@dataclass
class IndexManifest:
    version: int
    embedding_model: str
    reranker_model: str | None
    num_documents: int

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "IndexManifest":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


# =========================================================
# DOCUMENT MODEL
# =========================================================

@dataclass
class CaseDocument:
    id: str
    text: str
    metadata: dict[str, Any]


class DocumentStore:
    def __init__(self):
        self.docs: list[CaseDocument] = []

    def add(self, doc: CaseDocument):
        self.docs.append(doc)

    def __len__(self):
        return len(self.docs)

    def save(self, path: Path):
        path.write_text(
            json.dumps([d.__dict__ for d in self.docs], indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "DocumentStore":
        store = cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        store.docs = [CaseDocument(**d) for d in raw]
        return store


# =========================================================
# RESULT
# =========================================================

@dataclass
class RetrievalResult:
    document: CaseDocument
    score: float
    source: str


# =========================================================
# DENSE RETRIEVER
# =========================================================

class DenseRetriever:

    def __init__(self, embedding_model: EmbeddingModel, documents: list[CaseDocument]):
        self.embedding_model = embedding_model
        self.documents = documents
        self.index = self._build()

    def _build(self):
        texts = [d.text for d in self.documents]

        embeddings = self.embedding_model.embed_documents(texts)
        embeddings = np.asarray(embeddings, dtype="float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        return index

    def search(self, query: str, k: int):
        
        q = self.embedding_model.embed_query(query)
        q = np.asarray(q, dtype="float32")[None, :]

        scores, idx = self.index.search(q, k)
        return list(zip(scores[0], idx[0]))


# =========================================================
# SPARSE RETRIEVER
# =========================================================

class SparseRetriever:

    def __init__(self, documents: list[CaseDocument]):
        self.documents = documents
        tokenized = [d.text.lower().split() for d in documents]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        idx = np.argsort(scores)[::-1][:k]
        return [(scores[i], i) for i in idx]


# =========================================================
# RRF
# =========================================================

class RRF:

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, dense, sparse):
        scores = defaultdict(float)

        for rank, (_, i) in enumerate(dense):
            scores[i] += 1 / (self.k + rank + 1)

        for rank, (_, i) in enumerate(sparse):
            scores[i] += 1 / (self.k + rank + 1)

        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# =========================================================
# HYBRID INDEX
# =========================================================

class HybridCaseIndex:

    VERSION = 1

    def __init__(self, embedding_model: EmbeddingModel, reranker=None):
        self.embedding_model = embedding_model
        self.reranker = reranker

        self.store = DocumentStore()
        self.dense: DenseRetriever | None = None
        self.sparse: SparseRetriever | None = None

    # -------------------------
    # BUILD
    # -------------------------
    def build(self, documents: list[CaseDocument]):

        for doc in documents:
            self.store.add(doc)

        self.dense = DenseRetriever(self.embedding_model, self.store.docs)
        self.sparse = SparseRetriever(self.store.docs)

        logger.info("Built index with %d docs", len(self.store))

    # -------------------------
    # QUERY
    # -------------------------
    def query(self, text: str, k: int = 5):

        if not self.dense or not self.sparse:
            return []

        dense = self.dense.search(text, k * 3)
        sparse = self.sparse.search(text, k * 3)

        fused = RRF().fuse(dense, sparse)

        return [
            RetrievalResult(
                document=self.store.docs[i],
                score=score,
                source="fusion",
            )
            for i, score in fused[:k]
        ]

    # -------------------------
    # SAVE
    # -------------------------
    def save(self, path: str):

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        import faiss

        faiss.write_index(self.dense.index, str(path / "dense.faiss"))
        self.store.save(path / "documents.json")

        IndexManifest(
            version=self.VERSION,
            embedding_model=str(self.embedding_model),
            reranker_model=str(self.reranker) if self.reranker else None,
            num_documents=len(self.store),
        ).save(path / "manifest.json")

        logger.info("Saved index to %s", path)

    # -------------------------
    # LOAD
    # -------------------------
    @classmethod
    def load(cls, path: str, embedding_model: EmbeddingModel):

        path = Path(path)

        manifest = IndexManifest.load(path / "manifest.json")
        store = DocumentStore.load(path / "documents.json")

        import faiss

        obj = cls(embedding_model=embedding_model)
        obj.store = store

        obj.dense = DenseRetriever(embedding_model, store.docs)
        obj.dense.index = faiss.read_index(str(path / "dense.faiss"))

        obj.sparse = SparseRetriever(store.docs)

        logger.info(
            "Loaded v%d with %d docs",
            manifest.version,
            manifest.num_documents,
        )

        return obj

    # -------------------------
    # SOURCE BUILD
    # -------------------------
    @classmethod
    def from_source(cls, source_dir: str | Path, embedding_model: EmbeddingModel):

        source_dir = Path(source_dir)

        case_files: dict[str, list[Path]] = defaultdict(list)

        for file_path in source_dir.glob("*.txt"):
            case_id = file_path.name.split("_", 1)[0]
            case_files[case_id].append(file_path)

        docs: list[CaseDocument] = []

        for case_id, files in case_files.items():

            parts = []

            for fp in files:
                try:
                    parts.append(fp.read_text(encoding="utf-8").strip())
                except UnicodeDecodeError:
                    parts.append(fp.read_text(encoding="latin-1").strip())

            text = "\n\n".join([p for p in parts if p])

            if not text:
                continue

            docs.append(
                CaseDocument(
                    id=case_id,
                    text=text,
                    metadata={"case_number": case_id, "num_files": len(files)},
                )
            )

        index = cls(embedding_model)
        index.build(docs)

        return index

    # -------------------------
    # UTIL
    # -------------------------
    @property
    def size(self) -> int:
        return len(self.store)


print("INDEX MODULE LOADED")
print("HybridCaseIndex in globals:", "HybridCaseIndex" in globals())