"""Shared dependencies for the API — model loading, feature extraction, index."""

from __future__ import annotations

import logging
from typing import Any

from counterfactual.analyzer import CounterfactualAnalyzer
from features.config import FeaturesConfig
from features.extraction import FeatureExtractor
from models.config import MLflowConfig
from models.tracking import load_production_model
from retrieval.config import RetrievalConfig
from retrieval.index import CaseIndex

logger = logging.getLogger(__name__)


class AppState:
    """Holds loaded models and services for the API lifetime."""

    def __init__(self) -> None:
        self.classifier: Any = None
        self.regressor: Any = None
        self.feature_extractor: FeatureExtractor | None = None
        self.case_index: CaseIndex | None = None
        self.counterfactual_analyzer: CounterfactualAnalyzer | None = None
        self.models_loaded: bool = False

    def load_models(self, mlflow_config: MLflowConfig | None = None) -> None:
        """Load production models from MLflow registry."""
        config = mlflow_config or MLflowConfig()
        try:
            self.classifier = load_production_model(config.classifier_model_name, config)
            self.regressor = load_production_model(config.regressor_model_name, config)
            self.counterfactual_analyzer = CounterfactualAnalyzer(self.classifier, self.regressor)
            self.models_loaded = True
            logger.info("Production models loaded successfully")
        except Exception:
            logger.exception("Failed to load production models")
            self.models_loaded = False

    def load_feature_extractor(self, config: FeaturesConfig | None = None) -> None:
        self.feature_extractor = FeatureExtractor(config or FeaturesConfig())

    def load_case_index(self, config: RetrievalConfig | None = None) -> None:
        config = config or RetrievalConfig()
        self.case_index = CaseIndex(config)
        try:
            self.case_index.load()
            logger.info("Case index loaded: %d cases", self.case_index.size)
        except Exception:
            logger.warning("No existing case index found — retrieval will be unavailable")
            self.case_index = None


app_state = AppState()
