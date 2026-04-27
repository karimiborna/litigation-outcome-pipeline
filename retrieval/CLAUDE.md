# Retrieval Module

Embedding-based similar case retrieval and explanation generation grounded in historical examples.

## Responsibilities

- Generate embeddings for case text (using a sentence/document embedding model)
- Build and maintain a vector index of historical cases
- Given a new case, retrieve the top-K most similar historical cases
- Use retrieved cases to generate explanations that are grounded in real examples
- Provide context for predictions — "cases like yours typically resulted in..."

## Key Considerations

- Embedding model choice affects retrieval quality — needs evaluation
- Vector store options: FAISS, ChromaDB, Pinecone, or similar
- Retrieval is a complement to prediction, not a replacement
- Explanations must reference real cases, not hallucinated examples
- Index needs to be rebuilt/updated when new historical data is added
- Latency matters if used in real-time inference path
