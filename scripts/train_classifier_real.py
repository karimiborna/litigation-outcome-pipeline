#!/usr/bin/env python3
"""Train real classifier and regressor models from dataset.csv.

This is the production-style path for the local project demo. The notebook stays
as a smoke test; this script trains both API-required models through MLflow.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.config import MLflowConfig
from models.dataset import (
    MODEL_FEATURE_COLUMNS,
    PreparedDataset,
    dataset_sha256,
    load_dataset_csv,
    prepare_classifier_dataset,
    prepare_regressor_dataset,
)
from models.trainer import ClassifierTrainer, RegressorTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATASET_CSV = _REPO_ROOT / "dataset.csv"


def _audit_params(dataset: PreparedDataset, dataset_hash: str, model_kind: str) -> dict[str, Any]:
    return {
        "dataset_path": str(DATASET_CSV),
        "dataset_sha256": dataset_hash,
        "feature_version": "v2",
        "model_kind": model_kind,
        "raw_rows": dataset.raw_rows,
        "target_rows": dataset.target_rows,
        "model_rows": dataset.model_rows,
        "dropped_target_rows": dataset.dropped_target_rows,
        "dropped_feature_rows": dataset.dropped_feature_rows,
        "preprocessing": "drop_na_selected_features",
    }


def _write_feature_columns_artifact(directory: Path) -> Path:
    path = directory / "feature_columns_v2.json"
    path.write_text(json.dumps(list(MODEL_FEATURE_COLUMNS), indent=2), encoding="utf-8")
    return path


def _validate_classifier(dataset: PreparedDataset) -> None:
    if dataset.y.nunique() < 2:
        raise RuntimeError("Only one classifier class present after filtering.")
    class_counts = dataset.y.value_counts().to_dict()
    logger.info(
        "Classifier rows: %d, features: %d, class_counts=%s",
        len(dataset.x),
        dataset.x.shape[1],
        class_counts,
    )


def _validate_regressor(dataset: PreparedDataset) -> None:
    if len(dataset.x) < 5:
        raise RuntimeError("Too few regressor rows after filtering.")
    logger.info(
        "Regressor rows: %d, features: %d, target_mean=%.2f",
        len(dataset.x),
        dataset.x.shape[1],
        float(dataset.y.mean()),
    )


def main() -> int:
    config = MLflowConfig()
    logger.info("MLflow tracking URI: %s", config.tracking_uri)

    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Expected {DATASET_CSV} — run the dataset notebook first.")

    df = load_dataset_csv(DATASET_CSV)
    dataset_hash = dataset_sha256(DATASET_CSV)

    classifier_data = prepare_classifier_dataset(df)
    regressor_data = prepare_regressor_dataset(df)
    _validate_classifier(classifier_data)
    _validate_regressor(regressor_data)

    clf_trainer = ClassifierTrainer(config)
    reg_trainer = RegressorTrainer(config)

    with tempfile.TemporaryDirectory() as tmp:
        feature_columns_artifact = _write_feature_columns_artifact(Path(tmp))
        artifacts = [feature_columns_artifact, DATASET_CSV]

        logger.info("Training classifier...")
        clf_metrics = clf_trainer.train(
            classifier_data.x,
            classifier_data.y,
            run_name="real-dataset-v2",
            extra_params=_audit_params(classifier_data, dataset_hash, "classifier"),
            artifacts=artifacts,
        )
        logger.info("Classifier metrics: %s", clf_metrics)

        logger.info("Training regressor...")
        reg_metrics = reg_trainer.train(
            regressor_data.x,
            regressor_data.y,
            run_name="real-dataset-v2",
            extra_params=_audit_params(regressor_data, dataset_hash, "regressor"),
            artifacts=artifacts,
        )
        logger.info("Regressor metrics: %s", reg_metrics)

    print()
    print("Registered new model versions:")
    print(f"  - {config.classifier_model_name}")
    print(f"  - {config.regressor_model_name}")
    print()
    print("Promote both to Production with:")
    print("  python scripts/promote_models_to_production.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
