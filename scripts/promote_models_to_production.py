#!/usr/bin/env python3
"""Promote the latest registry version of each model to Production.

Validates the candidate versions via models.validation before flipping the
stage. Promotion is all-or-nothing: if either model fails any check, neither
is promoted, so the API is never left in a half-broken state.

The same validation suite runs at API startup (see api/dependencies.py), so a
model that the gate accepts is one the API will load successfully.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Allow `python scripts/promote_models_to_production.py` without editable install
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from models.config import MLflowConfig
from models.tracking import init_mlflow
from models.validation import (
    ModelValidationError,
    smoke_predict,
    validate_classifier,
    validate_regressor,
)


def _latest_version(client: MlflowClient, name: str) -> int:
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise ModelValidationError(f"No registered versions found for model '{name}'")
    return max(int(v.version) for v in versions)


def _load_candidate(name: str, version: int) -> Any:
    return mlflow.sklearn.load_model(f"models:/{name}/{version}")


def main() -> int:
    config = MLflowConfig()
    client = init_mlflow(config)

    classifier_version = _latest_version(client, config.classifier_model_name)
    regressor_version = _latest_version(client, config.regressor_model_name)

    print(f"Validating candidate {config.classifier_model_name} v{classifier_version}...")
    classifier = _load_candidate(config.classifier_model_name, classifier_version)
    validate_classifier(classifier)

    print(f"Validating candidate {config.regressor_model_name} v{regressor_version}...")
    regressor = _load_candidate(config.regressor_model_name, regressor_version)
    validate_regressor(regressor)

    print("Running smoke prediction...")
    smoke_predict(classifier, regressor)

    print("All checks passed. Promoting to Production.")
    client.transition_model_version_stage(
        name=config.classifier_model_name,
        version=classifier_version,
        stage="Production",
    )
    client.transition_model_version_stage(
        name=config.regressor_model_name,
        version=regressor_version,
        stage="Production",
    )
    print(f"Production: {config.classifier_model_name} v{classifier_version}")
    print(f"Production: {config.regressor_model_name} v{regressor_version}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ModelValidationError as exc:
        print(f"Promotion blocked: {exc}", file=sys.stderr)
        sys.exit(1)
