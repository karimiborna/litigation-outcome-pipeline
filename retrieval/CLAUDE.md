# Retrieval Module

Finds similar historical cases using sentence-transformers + FAISS.

## File Map

| File | Purpose |
|---|---|
| `embeddings.py` | EmbeddingModel — embed(), embed_batch(), L2-normalized |
| `index.py` | CaseIndex (FAISS), CaseMetadataStore, SimilarCaseResult |
| `config.py` | Embedding model, index path, top-K, similarity threshold |

## How It Works

1. At index build time: embed all case texts → add to FAISS flat IP index
2. At inference time: embed query text → search index → return top-K similar cases
3. Metadata store (JSON) maps index positions → case number, title, outcome

## Config Defaults

- Embedding model: `all-MiniLM-L6-v2` (sentence-transformers)
- Top-K: 5 similar cases
- Similarity threshold: 0.3 (inner product after L2 normalization = cosine similarity)
- Index path: `data/retrieval_index`

## Usage

```python
from retrieval.index import CaseIndex
from retrieval.config import RetrievalConfig

index = CaseIndex(RetrievalConfig())
index.build(case_texts, metadata_list)
index.save()

results = index.search(query_text, top_k=5)
# returns list of SimilarCaseResult(case_number, title, outcome, similarity_score)
```

## Purpose in Pipeline

Similar cases ground the LLM explanation in real examples. The API `/similar` endpoint returns these alongside predictions so users can see "here are 3 cases like yours and what happened."
