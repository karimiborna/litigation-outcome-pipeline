"""Model training pipeline for classification and regression."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import mlflow
from features.schema import FeatureVector
from models.config import MLflowConfig
from models.tracking import (
    get_or_create_experiment,
    log_metrics,
    log_model_artifact,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


def vectors_to_dataframe(vectors: list[FeatureVector]) -> pd.DataFrame:
    """Convert a list of FeatureVectors to a pandas DataFrame of model inputs."""
    rows = [v.to_model_input() for v in vectors]
    return pd.DataFrame(rows)


class ClassifierTrainer:
    """Trains and evaluates a binary classifier for plaintiff win prediction."""

    def __init__(self, config: MLflowConfig | None = None, **model_params: Any):
        self._config = config or MLflowConfig()
        self._params: dict[str, Any] = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": RANDOM_STATE,
        }
        self._params.update(model_params)
        self._model = GradientBoostingClassifier(**self._params)

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        test_size: float = 0.2,
        run_name: str | None = None,
    ) -> dict[str, float]:
        """Train the classifier and log everything to MLflow.

        Returns a dict of evaluation metrics.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=RANDOM_STATE, stratify=labels
        )

        experiment_name = self._config.classifier_experiment
        mlflow.set_tracking_uri(self._config.tracking_uri)
        experiment_id = get_or_create_experiment(experiment_name, self._config)

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            mlflow.log_params(self._params)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("n_samples", len(features))
            mlflow.log_param("n_features", features.shape[1])

            self._model.fit(x_train, y_train)

            y_pred = self._model.predict(x_test)
            y_proba = self._model.predict_proba(x_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_proba),
            }

            log_metrics(metrics)

            importances = dict(
                zip(features.columns, self._model.feature_importances_, strict=False)
            )
            top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat_name, importance in top_features:
                mlflow.log_metric(f"importance_{feat_name}", importance)

            log_model_artifact(
                self._model,
                artifact_path="classifier",
                registered_name=self._config.classifier_model_name,
            )

            logger.info("Classifier metrics: %s", metrics)
            return metrics

    @property
    def model(self) -> GradientBoostingClassifier:
        return self._model

    @property
    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_


class RegressorTrainer:
    """Trains and evaluates a regressor for expected monetary outcome prediction."""

    def __init__(self, config: MLflowConfig | None = None, **model_params: Any):
        self._config = config or MLflowConfig()
        self._params: dict[str, Any] = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "random_state": RANDOM_STATE,
        }
        self._params.update(model_params)
        self._model = GradientBoostingRegressor(**self._params)

    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        test_size: float = 0.2,
        run_name: str | None = None,
    ) -> dict[str, float]:
        """Train the regressor and log everything to MLflow.

        Returns a dict of evaluation metrics.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=RANDOM_STATE
        )

        experiment_name = self._config.regressor_experiment
        mlflow.set_tracking_uri(self._config.tracking_uri)
        experiment_id = get_or_create_experiment(experiment_name, self._config)

        with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
            mlflow.log_params(self._params)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("n_samples", len(features))
            mlflow.log_param("n_features", features.shape[1])

            self._model.fit(x_train, y_train)

            y_pred = self._model.predict(x_test)

            metrics = {
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                "r2": r2_score(y_test, y_pred),
            }

            log_metrics(metrics)

            importances = dict(
                zip(features.columns, self._model.feature_importances_, strict=False)
            )
            top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
            for feat_name, importance in top_features:
                mlflow.log_metric(f"importance_{feat_name}", importance)

            log_model_artifact(
                self._model,
                artifact_path="regressor",
                registered_name=self._config.regressor_model_name,
            )

            logger.info("Regressor metrics: %s", metrics)
            return metrics

    @property
    def model(self) -> GradientBoostingRegressor:
        return self._model

    @property
    def feature_importances(self) -> np.ndarray:
        return self._model.feature_importances_
