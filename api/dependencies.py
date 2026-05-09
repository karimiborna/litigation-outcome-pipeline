"""Shared dependencies for the API — model loading, feature extraction, index."""

from __future__ import annotations

import logging
from typing import Any
from pathlib import Path
from counterfactual.analyzer import CounterfactualAnalyzer
from features.config import FeaturesConfig
from features.extraction import FeatureExtractor
from models.config import MLflowConfig
from models.tracking import init_mlflow, load_production_model
from models.validation import ModelValidationError, validate_all
from retrieval.config import RetrievalConfig
from retrieval.index import HybridCaseIndex
from retrieval.index import CaseDocument
from sentence_transformers import SentenceTransformer

from retrieval.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class AppState:
    """Holds loaded models and services for the API lifetime."""

    def __init__(self, mlflow_config: MLflowConfig | None = None) -> None:
        self.mlflow_config = mlflow_config or MLflowConfig()
        self.classifier: Any = None
        self.regressor: Any = None
        self.feature_extractor: FeatureExtractor | None = None
        self.case_index: HybridCaseIndex | None = None
        self.counterfactual_analyzer: CounterfactualAnalyzer | None = None
        self.models_loaded: bool = False
        self.classifier_loaded: bool = False
        self.regressor_loaded: bool = False

        # Initialize MLflow connection on startup
        try:
            init_mlflow(self.mlflow_config)
            logger.info("MLflow initialized with tracking URI: %s", self.mlflow_config.tracking_uri)
        except Exception as e:
            logger.warning("Failed to initialize MLflow: %s", e)

    def load_models(self, mlflow_config: MLflowConfig | None = None) -> None:
        """Load production models from MLflow registry.

        Loads the classifier and regressor sequentially on the main thread.
        MLflow's artifact downloader uses multiprocessing.fork() internally; on
        macOS, calling fork() from a worker thread crashes the child due to
        Objective-C / OpenMP fork-safety. Sequential loading on the main thread
        avoids that.
        """
        config = mlflow_config or self.mlflow_config
        self.classifier_loaded = False
        self.regressor_loaded = False
        self.models_loaded = False
        self.counterfactual_analyzer = None

        try:
            self.classifier = load_production_model(config.classifier_model_name, config)
            self.classifier_loaded = True
            logger.info("Classifier loaded from registry: %s", config.classifier_model_name)
        except Exception as e:
            logger.warning("Failed to load classifier: %s", e)
            self.classifier = None

        try:
            self.regressor = load_production_model(config.regressor_model_name, config)
            self.regressor_loaded = True
            logger.info("Regressor loaded from registry: %s", config.regressor_model_name)
        except Exception as e:
            logger.warning("Failed to load regressor: %s", e)
            self.regressor = None

        if not (self.classifier_loaded and self.regressor_loaded):
            if not self.classifier_loaded:
                logger.warning("Classifier not loaded — predictions will fail")
            if not self.regressor_loaded:
                logger.warning("Regressor not loaded — predictions will fail")
            return

        try:
            validate_all(self.classifier, self.regressor)
        except ModelValidationError as e:
            logger.error(
                "Production models failed startup validation: %s. "
                "Marking models_loaded=False so /predict returns 503 instead of "
                "serving wrong predictions. Fix and re-promote.",
                e,
            )
            return

        self.counterfactual_analyzer = CounterfactualAnalyzer(self.classifier, self.regressor)
        self.models_loaded = True
        logger.info("Production classifier and regressor ready (MLflow: %s)", config.tracking_uri)

    def load_feature_extractor(self, config: FeaturesConfig | None = None) -> None:
        try:
            self.feature_extractor = FeatureExtractor(config or FeaturesConfig())
            logger.info("Feature extractor initialized")
        except ImportError as e:
            logger.warning("Feature extractor dependencies missing (%s)", e)
            self.feature_extractor = None
        except Exception as e:
            logger.warning("Failed to initialize feature extractor: %s", e)
            self.feature_extractor = None

        if self.feature_extractor is None:
            logger.warning("Feature extractor not ready — feature extraction will fail")

    def load_case_index(self, config: RetrievalConfig | None = None) -> None:
        config = config or RetrievalConfig()
        embedding_model = EmbeddingModel(config)

        index_path = Path("data/retrieval_index")
        manifest_file = index_path / "manifest.json"

        try:
            if manifest_file.exists():
                # Load existing index
                logger.info("Loading existing hybrid case index from %s", index_path)
                self.case_index = HybridCaseIndex.load(
                    path=index_path,
                    embedding_model=embedding_model
                )
            else:
                # Build from source if index is missing
                logger.info("No existing index found — building from source")
                self.case_index = HybridCaseIndex.from_source(
                    source_dir="scraper/processed",
                    embedding_model=embedding_model,
                )
                logger.info("Built index with %d documents", len(self.case_index.store.docs))
                self.case_index.save(index_path)

            logger.info("Case index ready: %d cases", self.case_index.size)

        except ImportError as e:
            logger.warning("FAISS not available (%s) — case index unavailable", e)
            self.case_index = None
        except Exception as e:
            logger.warning("Failed to load case index: %s", e)
            self.case_index = None


app_state = AppState()
