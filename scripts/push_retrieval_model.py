#!/usr/bin/env python3
"""Push the RAG retrieval model (embedding model + FAISS index) to MLflow.

This script packages the sentence-transformers embedding model along with the
FAISS vector index and case metadata, and registers them as a single artifact
in the MLflow Model Registry.

Usage:
    python scripts/push_retrieval_model.py [--index-dir DATA/retrieval_index]
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

from retrieval.config import RetrievalConfig
from retrieval.embeddings import EmbeddingModel
from retrieval.index import CaseIndex

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MLFLOW_MODEL_NAME = "litigation-retrieval-rag"


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Push the RAG retrieval model to MLflow.")
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/retrieval_index"),
        help="Directory containing the FAISS index and metadata",
    )
    args = parser.parse_args()

    index_dir = args.index_dir
    if not index_dir.exists():
        raise SystemExit(f"Index directory not found: {index_dir}\n"
                         "Run 'python scripts/build_retrieval_index.py' first.")

    if not (index_dir / "index.faiss").exists():
        raise SystemExit(f"FAISS index not found: {index_dir / 'index.faiss'}")
    if not (index_dir / "metadata.json").exists():
        raise SystemExit(f"Metadata not found: {index_dir / 'metadata.json'}")

    # Import mlflow here so the script fails fast if it's not installed
    import mlflow
    from models.config import MLflowConfig

    mlflow_config = MLflowConfig()
    logger.info(f"MLflow tracking URI: http://35.208.251.175:5000/")
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)

    config = RetrievalConfig()

    # Wrap the index + embedding model in a custom MLflow pyfunc model
    from mlflow.pyfunc import PythonModel, PythonModelContext
    import numpy as np
    import faiss

    class RetrievalRAGModel(PythonModel):
        """MLflow pyfunc wrapper for the RAG retrieval system.

        The model accepts a query string and returns a list of similar case dicts.
        """

        def load_context(self, context: PythonModelContext) -> None:
            import json

            # Load the FAISS index
            index_path = context.artifacts["faiss_index"]
            self._index = faiss.read_index(index_path)

            # Load metadata
            meta_path = context.artifacts["metadata"]
            with open(meta_path) as f:
                self._metadata = json.load(f)

            # Load embedding model
            self._embedding_model = SentenceTransformer(
                config.embedding_model,
                device="cpu",
            )
            logger.info(
                "Loaded RAG model: index_size=%d, metadata_count=%d, embedding_dim=%d",
                self._index.ntotal,
                len(self._metadata),
                self._embedding_model.get_sentence_embedding_dimension(),
            )

        def predict(self, context, model_input, params=None):
            """Predict similar cases for each query string.

            Args:
                model_input: A string query or a list/DataFrame/Series of query strings.
            Returns:
                List of lists of similar case dicts.
            """
            # Normalize input to a list of strings
            if isinstance(model_input, str):
                queries = [model_input]
            elif hasattr(model_input, "tolist"):
                queries = model_input.tolist()
            else:
                queries = list(model_input)

            top_k = (params or {}).get("top_k", config.top_k)
            threshold = (params or {}).get("similarity_threshold", config.similarity_threshold)

            # Generate embeddings
            embeddings = self._embedding_model.encode(
                queries,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if isinstance(embeddings, np.ndarray):
                query_vectors = embeddings.astype(np.float32)
            else:
                query_vectors = np.array(embeddings, dtype=np.float32)

            # Search the index
            if query_vectors.ndim == 1:
                query_vectors = query_vectors.reshape(1, -1)

            scores, indices = self._index.search(query_vectors, min(top_k, self._index.ntotal))

            results = []
            for query_scores, query_indices in zip(scores, indices):
                query_results = []
                for score, idx in zip(query_scores, query_indices):
                    if idx < 0:
                        continue
                    if score < threshold:
                        continue
                    meta = self._metadata[idx]
                    query_results.append({
                        "case_number": meta["case_number"],
                        "case_title": meta["case_title"],
                        "similarity_score": round(float(score), 4),
                        "outcome": meta.get("outcome"),
                        "case_snippet": meta.get("case_snippet"),
                    })
                results.append(query_results)

            # If single input, return single list (not wrapped)
            if isinstance(model_input, str):
                return results[0]
            return results

    # Build the MLflow model
    from sentence_transformers import SentenceTransformer

    logger.info("Building RAG model package...")
    logger.info("  Index directory: %s", index_dir)
    logger.info("  Embedding model: %s", config.embedding_model)

    # Read index size for logging
    test_index = faiss.read_index(str(index_dir / "index.faiss"))
    logger.info("  FAISS index: %d vectors, dimension %d", test_index.ntotal, test_index.d)

    # Create the model signature
    from mlflow.models.signature import infer_signature
    import pandas as pd

    sample_input = ["sample query"]
    sample_output = [[]]  # Empty results for sample
    signature = infer_signature(
        pd.DataFrame({"input": sample_input}),
        sample_output,
    )

    # Log the model
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Copy index files to temp dir for artifact logging
        import shutil
        tmp_index_dir = Path(tmp_dir) / "retrieval_index"
        tmp_index_dir.mkdir()
        shutil.copy(index_dir / "index.faiss", tmp_index_dir / "index.faiss")
        shutil.copy(index_dir / "metadata.json", tmp_index_dir / "metadata.json")

        # Also save the embedding model itself
        embedding_model = SentenceTransformer(config.embedding_model, device="cpu")
        embedding_model_path = Path(tmp_dir) / "embedding_model"
        embedding_model.save(str(embedding_model_path))

        with mlflow.start_run(run_name="retrieval-rag") as run:
            mlflow.log_param("embedding_model", config.embedding_model)
            mlflow.log_param("embedding_dim", embedding_model.get_sentence_embedding_dimension())
            mlflow.log_param("index_size", test_index.ntotal)
            mlflow.log_param("faiss_dimension", test_index.d)
            mlflow.log_param("top_k", config.top_k)
            mlflow.log_param("similarity_threshold", config.similarity_threshold)

            mlflow.pyfunc.log_model(
                artifact_path="retrieval-rag",
                python_model=RetrievalRAGModel(),
                artifacts={
                    "faiss_index": str(tmp_index_dir / "index.faiss"),
                    "metadata": str(tmp_index_dir / "metadata.json"),
                },
                signature=signature,
                registered_model_name=_MLFLOW_MODEL_NAME,
            )

            logger.info("Model logged to MLflow run: %s", run.info.run_id)

    logger.info("")
    logger.info("Registered new model version of '%s'", _MLFLOW_MODEL_NAME)
    logger.info("")
    logger.info("View at: %s/#/models/%s", mlflow_config.tracking_uri.rstrip("/"), _MLFLOW_MODEL_NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())