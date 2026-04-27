"""Tests for dataset.csv preprocessing shared by training and inference."""

from __future__ import annotations

import pandas as pd
import pytest

from features.schema import FeatureVector
from models.dataset import (
    MODEL_FEATURE_COLUMNS,
    RAW_MODEL_FEATURE_COLUMNS,
    feature_vector_to_model_frame,
    prepare_classifier_dataset,
    prepare_regressor_dataset,
)


def _row(**overrides):
    row = {col: False for col in RAW_MODEL_FEATURE_COLUMNS}
    row.update(
        {
            "feat_user_is_plaintiff": True,
            "feat_claim_category": "unpaid_debt",
            "feat_monetary_amount_claimed": 1000.0,
            "feat_claim_amount_stated_in_dollars": True,
            "feat_plaintiff_count": 1.0,
            "feat_defendant_count": 1.0,
            "feat_witness_count": 0,
            "feat_text_length": 500,
            "feat_document_count": 2,
            "label_outcome": "plaintiff_win",
            "label_total_awarded": 800.0,
        }
    )
    row.update(overrides)
    return row


def test_classifier_dataset_drops_missing_targets_and_features() -> None:
    df = pd.DataFrame(
        [
            _row(label_outcome="plaintiff_win"),
            _row(label_outcome="dismissed"),
            _row(label_outcome=None),
            _row(feat_monetary_amount_claimed=None),
        ]
    )

    prepared = prepare_classifier_dataset(df)

    assert prepared.raw_rows == 4
    assert prepared.target_rows == 3
    assert prepared.model_rows == 2
    assert prepared.dropped_target_rows == 1
    assert prepared.dropped_feature_rows == 1
    assert prepared.y.tolist() == [1, 0]
    assert tuple(prepared.x.columns) == MODEL_FEATURE_COLUMNS
    assert not prepared.x.isna().any().any()


def test_regressor_dataset_drops_missing_targets_and_features() -> None:
    df = pd.DataFrame(
        [
            _row(label_total_awarded=800.0),
            _row(label_total_awarded=None),
            _row(feat_plaintiff_count=None),
        ]
    )

    prepared = prepare_regressor_dataset(df)

    assert prepared.model_rows == 1
    assert prepared.y.tolist() == [800.0]
    assert tuple(prepared.x.columns) == MODEL_FEATURE_COLUMNS


def test_feature_vector_to_model_frame_matches_training_columns() -> None:
    vector = FeatureVector(
        case_number="CSM123",
        feature_version="v2",
        user_is_plaintiff=True,
        claim_category="unpaid_debt",
        monetary_amount_claimed=1000.0,
        user_has_attorney=False,
        counterclaim_present=False,
        has_photos_or_physical_evidence=False,
        has_receipts_or_financial_records=True,
        has_written_communications=True,
        has_witness_statements=False,
        has_repair_or_replacement_estimate=False,
        has_police_report=False,
        has_medical_records=False,
        has_expert_assessment=False,
        has_invoices_or_billing_records=True,
        argument_cites_specific_dates=True,
        argument_cites_specific_dollar_amounts=True,
        argument_cites_contract_or_document=False,
        argument_has_chronological_timeline=True,
        argument_names_specific_witnesses=False,
        argument_quantifies_each_damage_component=True,
        argument_cites_statute_or_legal_basis=False,
        argument_identifies_specific_location=True,
        claim_amount_stated_in_dollars=True,
        plaintiff_count=1,
        defendant_count=1,
        witness_count=0,
        text_length=500,
        document_count=2,
    )

    frame = feature_vector_to_model_frame(vector)

    assert tuple(frame.columns) == MODEL_FEATURE_COLUMNS
    assert frame.loc[0, "feat_claim_category_unpaid_debt"] == 1.0
    assert not frame.isna().any().any()


def test_feature_vector_to_model_frame_rejects_missing_required_feature() -> None:
    vector = FeatureVector(case_number="CSM123", feature_version="v2")

    with pytest.raises(ValueError, match="Missing required inference features"):
        feature_vector_to_model_frame(vector)
