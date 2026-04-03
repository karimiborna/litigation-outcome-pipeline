"""MLflow and model training configuration."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class MLflowConfig(BaseSettings):
    """MLflow tracking and registry settings, loaded from env vars or .env."""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    tracking_uri: str = Field(
        default="http://localhost:5000",
        alias="MLFLOW_TRACKING_URI",
    )
    artifact_root: str = Field(
        default="mlruns/artifacts",
        alias="MLFLOW_ARTIFACT_ROOT",
    )
    registry_uri: str = Field(
        default="",
        alias="MLFLOW_REGISTRY_URI",
    )

    classifier_experiment: str = "litigation-classifier"
    regressor_experiment: str = "litigation-regressor"

    classifier_model_name: str = "litigation-win-classifier"
    regressor_model_name: str = "litigation-monetary-regressor"

    @property
    def effective_registry_uri(self) -> str:
        return self.registry_uri or self.tracking_uri
