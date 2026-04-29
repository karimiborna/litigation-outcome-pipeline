"""Shared dependencies for the API — model loading, feature extraction, index."""

from __future__ import annotations

import logging
from typing import Any

from counterfactual.analyzer import CounterfactualAnalyzer
from features.config import FeaturesConfig
from features.extraction import FeatureExtractor
from models.config import MLflowConfig
from models.tracking import init_mlflow, load_production_model
from retrieval.config import RetrievalConfig
from retrieval.index import CaseIndex

logger = logging.getLogger(__name__)


class AppState:
    """Holds loaded models and services for the API lifetime."""

    def __init__(self, mlflow_config: MLflowConfig | None = None) -> None:
        self.mlflow_config = mlflow_config or MLflowConfig()
        self.classifier: Any = None
        self.regressor: Any = None
        self.feature_extractor: FeatureExtractor | None = None
        self.case_index: CaseIndex | None = None
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
        """Load production models from MLflow registry."""
        import threading

        config = mlflow_config or self.mlflow_config
        self.classifier_loaded = False
        self.regressor_loaded = False
        self.models_loaded = False
        self.counterfactual_analyzer = None

        def load_classifier_task():
            try:
                self.classifier = load_production_model(config.classifier_model_name, config)
                self.classifier_loaded = True
                logger.info("Classifier loaded from registry: %s", config.classifier_model_name)
            except Exception as e:
                logger.warning("Failed to load classifier: %s", e)
                self.classifier = None

        def load_regressor_task():
            try:
                self.regressor = load_production_model(config.regressor_model_name, config)
                self.regressor_loaded = True
                logger.info("Regressor loaded from registry: %s", config.regressor_model_name)
            except Exception as e:
                logger.warning("Failed to load regressor: %s", e)
                self.regressor = None

        # Load models in parallel with timeout
        classifier_thread = threading.Thread(target=load_classifier_task, daemon=True)
        regressor_thread = threading.Thread(target=load_regressor_task, daemon=True)

        classifier_thread.start()
        regressor_thread.start()

        # Wait maximum 30 seconds for both to finish
        classifier_thread.join(timeout=30)
        regressor_thread.join(timeout=30)

        if self.classifier_loaded and self.regressor_loaded:
            self.counterfactual_analyzer = CounterfactualAnalyzer(self.classifier, self.regressor)
            self.models_loaded = True
            logger.info(
                "Production classifier and regressor ready (MLflow: %s)", config.tracking_uri
            )
        else:
            if not self.classifier_loaded:
                logger.warning("Classifier not loaded — predictions will fail")
            if not self.regressor_loaded:
                logger.warning("Regressor not loaded — predictions will fail")

    def load_feature_extractor(self, config: FeaturesConfig | None = None) -> None:
        import threading

        def load_task():
            try:
                self.feature_extractor = FeatureExtractor(config or FeaturesConfig())
                logger.info("Feature extractor initialized")
            except ImportError as e:
                logger.warning("Feature extractor dependencies missing (%s)", e)
                self.feature_extractor = None
            except Exception as e:
                logger.warning("Failed to initialize feature extractor: %s", e)
                self.feature_extractor = None

        # Load in background thread with 15-second timeout
        thread = threading.Thread(target=load_task, daemon=True)
        thread.start()
        thread.join(timeout=15)

        if self.feature_extractor is None:
            logger.warning("Feature extractor not ready — feature extraction will fail")

    def load_case_index(self, config: RetrievalConfig | None = None) -> None:
        config = config or RetrievalConfig()
        try:
            self.case_index = CaseIndex(config)
            self.case_index.load()
            logger.info("Case index loaded: %d cases", self.case_index.size)
        except ImportError as e:
            logger.warning("FAISS not available (%s) — case index unavailable", e)
            self.case_index = None
        except Exception as e:
            logger.warning("Failed to load case index: %s", e)
            self.case_index = None


app_state = AppState()
