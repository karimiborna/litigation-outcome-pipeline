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


_BOOL_COLUMNS = [
    "user_is_plaintiff",
    "user_has_attorney",
    "opposing_party_has_attorney",
    "opposing_party_filed_response_documents",
    "counterclaim_present",
    "contract_present",
    "has_photos_or_physical_evidence",
    "has_receipts_or_financial_records",
    "has_written_communications",
    "has_witness_statements",
    "has_signed_contract_attached",
    "has_repair_or_replacement_estimate",
    "has_police_report",
    "has_medical_records",
    "has_expert_assessment",
    "has_invoices_or_billing_records",
    "argument_cites_specific_dates",
    "argument_cites_specific_dollar_amounts",
    "argument_cites_contract_or_document",
    "argument_has_chronological_timeline",
    "argument_names_specific_witnesses",
    "argument_quantifies_each_damage_component",
    "argument_cites_statute_or_legal_basis",
    "argument_identifies_specific_location",
    "sent_written_demand_letter",
    "sent_certified_mail",
    "gave_opportunity_to_cure",
    "attempted_mediation",
    "contract_is_written",
    "contract_is_signed_by_both_parties",
    "contract_specifies_deadline_or_term",
    "contract_specifies_payment_amount",
    "damages_include_out_of_pocket_costs",
    "damages_include_lost_wages",
    "damages_include_property_value_loss",
    "damages_are_ongoing",
    "damages_have_third_party_valuation",
    "claim_amount_stated_in_dollars",
    "claim_amount_is_within_small_claims_limit",
    "user_seeks_interest",
    "user_seeks_court_costs",
]


def _feature_columns() -> list[str]:
    return list(FeatureVector(case_number="TEMPLATE").to_model_input().keys())


def build_synthetic_dataset(
    n_samples: int = N_SAMPLES,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y_class, y_reg) with v2 feature column names for the pipeline."""
    rng = np.random.default_rng(random_state)
    cols = _feature_columns()
    x = pd.DataFrame(
        {c: rng.normal(0, 1, n_samples) for c in cols},
    )

    # Numeric ranges
    x["monetary_amount_claimed"] = rng.uniform(100, 15000, n_samples)
    x["witness_count"] = rng.integers(0, 5, n_samples).astype(float)
    x["plaintiff_count"] = rng.integers(1, 3, n_samples).astype(float)
    x["defendant_count"] = rng.integers(1, 4, n_samples).astype(float)
    x["text_length"] = rng.uniform(200, 8000, n_samples)
    x["document_count"] = rng.integers(1, 12, n_samples).astype(float)

    # Booleans (0/1)
    for b in _BOOL_COLUMNS:
        x[b] = rng.integers(0, 2, n_samples).astype(float)

    # Synthetic win signal: a few evidence/procedural booleans push win probability.
    logit = (
        0.25 * x["has_signed_contract_attached"]
        + 0.20 * x["has_receipts_or_financial_records"]
        + 0.15 * x["has_written_communications"]
        + 0.15 * x["sent_written_demand_letter"]
        + 0.10 * x["argument_cites_specific_dates"]
        + 0.10 * x["argument_quantifies_each_damage_component"]
        - 0.10 * x["opposing_party_has_attorney"]
        + 0.25 * (1.0 - x["opposing_party_filed_response_documents"])  # default risk
        + rng.normal(0, 0.5, n_samples)
    )
    y_class = (logit > 0).astype(int)
    y_reg = (
        0.3 * x["monetary_amount_claimed"]
        + 200.0 * x["has_signed_contract_attached"]
        + 150.0 * x["has_receipts_or_financial_records"]
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
