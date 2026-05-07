"""Shared dependencies for the API — model loading, feature extraction, index."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from counterfactual.analyzer import CounterfactualAnalyzer
from features.config import FeaturesConfig
from features.extraction import FeatureExtractor
from models.config import MLflowConfig
from models.tracking import load_production_model
from retrieval.config import RetrievalConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


class AppState:
    """Holds loaded models and services for the API lifetime."""

    def __init__(self) -> None:
        self.classifier: Any = None
        self.regressor: Any = None
        self.feature_extractor: FeatureExtractor | None = None
        self.case_index: Any = None
        self.counterfactual_analyzer: CounterfactualAnalyzer | None = None
        self.models_loaded: bool = False
        self.classifier_loaded: bool = False
        self.regressor_loaded: bool = False

    def load_models(self, mlflow_config: MLflowConfig | None = None) -> None:
        """Load production models from MLflow registry."""
        if _env_flag("API_SKIP_MODEL_LOADING"):
            logger.info("Skipping model loading because API_SKIP_MODEL_LOADING is enabled")
            self.classifier = None
            self.regressor = None
            self.counterfactual_analyzer = None
            self.classifier_loaded = False
            self.regressor_loaded = False
            self.models_loaded = False
            return

        config = mlflow_config or MLflowConfig()
        self.classifier_loaded = False
        self.regressor_loaded = False
        self.models_loaded = False
        self.counterfactual_analyzer = None

        try:
            self.classifier = load_production_model(config.classifier_model_name, config)
            self.classifier_loaded = True
            logger.info("Classifier loaded from registry: %s", config.classifier_model_name)
        except Exception:
            logger.exception("Failed to load classifier from MLflow registry")
            self.classifier = None

        try:
            self.regressor = load_production_model(config.regressor_model_name, config)
            self.regressor_loaded = True
            logger.info("Regressor loaded from registry: %s", config.regressor_model_name)
        except Exception:
            logger.exception("Failed to load regressor from MLflow registry")
            self.regressor = None

        if self.classifier_loaded and self.regressor_loaded:
            self.counterfactual_analyzer = CounterfactualAnalyzer(self.classifier, self.regressor)
            self.models_loaded = True
            logger.info("Production classifier and regressor ready")

    def load_feature_extractor(self, config: FeaturesConfig | None = None) -> None:
        self.feature_extractor = FeatureExtractor(config or FeaturesConfig())

    def load_case_index(self, config: RetrievalConfig | None = None) -> None:
        config = config or RetrievalConfig()
        index_path = Path(config.index_path)
        index_file = index_path / "index.faiss"
        metadata_file = index_path / "metadata.json"

        if not index_file.exists() or not metadata_file.exists():
            logger.info("No retrieval index found at %s — retrieval will be unavailable", index_path)
            self.case_index = None
            return

        from retrieval.index import CaseIndex

        self.case_index = CaseIndex(config)
        try:
            self.case_index.load()
            logger.info("Case index loaded: %d cases", self.case_index.size)
        except Exception:
            logger.warning("No existing case index found — retrieval will be unavailable")
            self.case_index = None


app_state = AppState()
