#!/usr/bin/env python3
"""Train the plaintiff-win classifier (and monetary regressor) on synthetic data.

Logs metrics and registers sklearn models with MLflow. After training, promote
versions to *Production* so the API can load them via the Model Registry:

    python scripts/promote_models_to_production.py

Requires a reachable MLflow tracking server with model registry support.
Point MLFLOW_TRACKING_URI at the hosted server before running:

    export MLFLOW_TRACKING_URI=http://35.208.251.175:5000
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow `python scripts/train_binary_classifier.py` without editable install
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from features.schema import FeatureVector
from models.config import MLflowConfig
from models.trainer import ClassifierTrainer, RegressorTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
N_SAMPLES = 400


def _feature_columns() -> list[str]:
    return list(
        FeatureVector(
            case_number="TEMPLATE",
            evidence_strength=3,
            contract_present=True,
            argument_clarity_plaintiff=3,
            argument_clarity_defendant=3,
            monetary_amount_claimed=1000.0,
            prior_attempts_to_resolve=True,
            witness_count=0,
            documentary_evidence=True,
            timeline_clarity=3,
            legal_representation_plaintiff=False,
            legal_representation_defendant=False,
            counterclaim_present=False,
            default_judgment_likely=False,
            plaintiff_count=1,
            defendant_count=1,
            has_attorney_plaintiff=False,
            has_attorney_defendant=False,
            text_length=500,
            document_count=3,
        ).to_model_input().keys()
    )


def build_synthetic_dataset(
    n_samples: int = N_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y_class, y_reg) with realistic column names for the pipeline."""
    rng = np.random.default_rng(random_state)
    cols = _feature_columns()
    x = pd.DataFrame(
        {c: rng.normal(0, 1, n_samples) for c in cols},
    )
    # Override a few columns to mimic real ranges
    x["evidence_strength"] = rng.integers(1, 6, n_samples)
    x["argument_clarity_plaintiff"] = rng.integers(1, 6, n_samples)
    x["argument_clarity_defendant"] = rng.integers(1, 6, n_samples)
    x["timeline_clarity"] = rng.integers(1, 6, n_samples)
    x["witness_count"] = rng.integers(0, 5, n_samples)
    x["monetary_amount_claimed"] = rng.uniform(100, 15000, n_samples)
    x["text_length"] = rng.uniform(200, 8000, n_samples)
    x["document_count"] = rng.integers(1, 12, n_samples)
    for b in (
        "contract_present",
        "prior_attempts_to_resolve",
        "documentary_evidence",
        "legal_representation_plaintiff",
        "legal_representation_defendant",
        "counterclaim_present",
        "default_judgment_likely",
        "has_attorney_plaintiff",
        "has_attorney_defendant",
    ):
        x[b] = rng.integers(0, 2, n_samples).astype(float)

    x["plaintiff_count"] = rng.integers(1, 3, n_samples).astype(float)
    x["defendant_count"] = rng.integers(1, 4, n_samples).astype(float)

    logit = (
        0.15 * x["evidence_strength"]
        + 0.12 * x["argument_clarity_plaintiff"]
        - 0.1 * x["argument_clarity_defendant"]
        + 0.08 * x["documentary_evidence"]
        - 0.05 * x["legal_representation_defendant"]
        + 0.2 * x["default_judgment_likely"]
        + rng.normal(0, 0.5, n_samples)
    )
    y_class = (logit > 0).astype(int)
    y_reg = (
        0.3 * x["monetary_amount_claimed"]
        + 50.0 * x["evidence_strength"]
        + rng.normal(0, 200, n_samples)
    ).clip(0, 25000)

    return x.astype(np.float64), pd.Series(y_class), pd.Series(y_reg)


def main() -> int:
    config = MLflowConfig()
    logger.info("MLflow tracking URI: %s", config.tracking_uri)

    x, y_class, y_reg = build_synthetic_dataset()
    if y_class.nunique() < 2:
        raise RuntimeError("Synthetic labels collapsed to one class; re-run with a different seed.")
    clf_trainer = ClassifierTrainer(config)
    reg_trainer = RegressorTrainer(config)

    logger.info("Training classifier (%d rows)...", len(x))
    clf_metrics = clf_trainer.train(x, y_class, run_name="synthetic-demo")
    logger.info("Classifier metrics: %s", clf_metrics)

    logger.info("Training regressor (%d rows)...", len(x))
    reg_metrics = reg_trainer.train(x, y_reg, run_name="synthetic-demo")
    logger.info("Regressor metrics: %s", reg_metrics)

    print()
    print("Registered models (new versions):")
    print(f"  - {config.classifier_model_name}")
    print(f"  - {config.regressor_model_name}")
    print()
    print("Promote the latest version of each to Production, then start the API:")
    print("  python scripts/promote_models_to_production.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
