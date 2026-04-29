"""Shared preprocessing for the real ``dataset.csv`` training table."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from features.schema import FeatureVector

WIN_OUTCOMES = ("plaintiff_win", "partial_win")

CLAIM_CATEGORIES = (
    "breach_of_contract",
    "fraud",
    "other",
    "personal_injury",
    "property_damage",
    "security_deposit",
    "service_dispute",
    "unpaid_debt",
)

RAW_MODEL_FEATURE_COLUMNS = (
    "feat_user_is_plaintiff",
    "feat_claim_category",
    "feat_monetary_amount_claimed",
    "feat_user_has_attorney",
    "feat_counterclaim_present",
    "feat_has_photos_or_physical_evidence",
    "feat_has_receipts_or_financial_records",
    "feat_has_written_communications",
    "feat_has_witness_statements",
    "feat_has_repair_or_replacement_estimate",
    "feat_has_police_report",
    "feat_has_medical_records",
    "feat_has_expert_assessment",
    "feat_has_invoices_or_billing_records",
    "feat_argument_cites_specific_dates",
    "feat_argument_cites_specific_dollar_amounts",
    "feat_argument_cites_contract_or_document",
    "feat_argument_has_chronological_timeline",
    "feat_argument_names_specific_witnesses",
    "feat_argument_quantifies_each_damage_component",
    "feat_argument_cites_statute_or_legal_basis",
    "feat_argument_identifies_specific_location",
    "feat_claim_amount_stated_in_dollars",
    "feat_plaintiff_count",
    "feat_defendant_count",
    "feat_witness_count",
    "feat_text_length",
    "feat_document_count",
    "feat_attempted_mediation",
    "feat_claim_amount_is_within_small_claims_limit",
    "feat_damages_are_ongoing",
    "feat_damages_have_third_party_valuation",
    "feat_damages_include_lost_wages",
    "feat_damages_include_out_of_pocket_costs",
    "feat_damages_include_property_value_loss",
    "feat_gave_opportunity_to_cure",
    "feat_has_signed_contract_attached",
    "feat_opposing_party_filed_response_documents",
    "feat_sent_certified_mail",
    "feat_sent_written_demand_letter",
    "feat_user_seeks_court_costs",
    "feat_user_seeks_interest",
    "feat_contract_present",
    "feat_opposing_party_has_attorney",
)

MODEL_FEATURE_COLUMNS = tuple(
    c for c in RAW_MODEL_FEATURE_COLUMNS if c != "feat_claim_category"
) + tuple(f"feat_claim_category_{c}" for c in CLAIM_CATEGORIES)


@dataclass(frozen=True)
class PreparedDataset:
    """Preprocessed model matrix, target, and useful audit metadata."""

    x: pd.DataFrame
    y: pd.Series
    raw_rows: int
    target_rows: int
    model_rows: int
    dropped_target_rows: int
    dropped_feature_rows: int
    feature_columns: tuple[str, ...]


def dataset_sha256(path: Path) -> str:
    """Return a stable hash for a dataset file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset_csv(path: Path) -> pd.DataFrame:
    """Load the joined feature/label dataset."""
    return pd.read_csv(path)


def preprocess_feature_frame(
    raw_features: pd.DataFrame,
    *,
    drop_missing: bool = False,
) -> tuple[pd.DataFrame, pd.Index]:
    """Convert raw ``feat_*`` columns to the model matrix.

    Rows with missing values in the selected raw features are dropped by default,
    matching the current project decision to avoid imputation.
    """
    missing_columns = [c for c in RAW_MODEL_FEATURE_COLUMNS if c not in raw_features.columns]
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")

    selected = raw_features.loc[:, list(RAW_MODEL_FEATURE_COLUMNS)].copy()
    before_index = selected.index
    if drop_missing:
        selected = selected.dropna(axis=0, how="any")

    category = pd.Categorical(
        selected["feat_claim_category"],
        categories=list(CLAIM_CATEGORIES),
    )
    category_dummies = pd.get_dummies(
        category,
        prefix="feat_claim_category",
        dtype=float,
    )
    category_dummies.index = selected.index

    numeric = selected.drop(columns=["feat_claim_category"]).astype(float)
    x = pd.concat([numeric, category_dummies], axis=1)
    x = x.reindex(columns=list(MODEL_FEATURE_COLUMNS), fill_value=0.0)

    kept_index = x.index
    dropped_index = before_index.difference(kept_index)
    return x.reset_index(drop=True), dropped_index


def prepare_classifier_dataset(df: pd.DataFrame) -> PreparedDataset:
    """Build the classifier matrix and binary plaintiff-win target."""
    raw_rows = len(df)
    target_df = df.dropna(subset=["label_outcome"]).copy()
    y = target_df["label_outcome"].isin(WIN_OUTCOMES).astype(int)

    x, dropped_index = preprocess_feature_frame(target_df, drop_missing=False)
    if len(dropped_index) > 0:
        y = y.drop(index=dropped_index)

    y = y.reset_index(drop=True)
    return PreparedDataset(
        x=x,
        y=y,
        raw_rows=raw_rows,
        target_rows=len(target_df),
        model_rows=len(x),
        dropped_target_rows=raw_rows - len(target_df),
        dropped_feature_rows=len(target_df) - len(x),
        feature_columns=MODEL_FEATURE_COLUMNS,
    )


def prepare_regressor_dataset(df: pd.DataFrame) -> PreparedDataset:
    """Build the regressor matrix and monetary-award target."""
    raw_rows = len(df)
    target_df = df.dropna(subset=["label_total_awarded"]).copy()
    y = target_df["label_total_awarded"].astype(float)

    x, dropped_index = preprocess_feature_frame(target_df, drop_missing=False)
    if len(dropped_index) > 0:
        y = y.drop(index=dropped_index)

    y = y.reset_index(drop=True)
    return PreparedDataset(
        x=x,
        y=y,
        raw_rows=raw_rows,
        target_rows=len(target_df),
        model_rows=len(x),
        dropped_target_rows=raw_rows - len(target_df),
        dropped_feature_rows=len(target_df) - len(x),
        feature_columns=MODEL_FEATURE_COLUMNS,
    )


def feature_vector_to_raw_row(vector: FeatureVector) -> dict[str, Any]:
    """Convert API-extracted features into the raw ``feat_*`` shape."""
    return {
        "feat_user_is_plaintiff": vector.user_is_plaintiff,
        "feat_claim_category": vector.claim_category,
        "feat_monetary_amount_claimed": vector.monetary_amount_claimed,
        "feat_user_has_attorney": vector.user_has_attorney,
        "feat_counterclaim_present": vector.counterclaim_present,
        "feat_has_photos_or_physical_evidence": vector.has_photos_or_physical_evidence,
        "feat_has_receipts_or_financial_records": vector.has_receipts_or_financial_records,
        "feat_has_written_communications": vector.has_written_communications,
        "feat_has_witness_statements": vector.has_witness_statements,
        "feat_has_repair_or_replacement_estimate": vector.has_repair_or_replacement_estimate,
        "feat_has_police_report": vector.has_police_report,
        "feat_has_medical_records": vector.has_medical_records,
        "feat_has_expert_assessment": vector.has_expert_assessment,
        "feat_has_invoices_or_billing_records": vector.has_invoices_or_billing_records,
        "feat_argument_cites_specific_dates": vector.argument_cites_specific_dates,
        "feat_argument_cites_specific_dollar_amounts": (
            vector.argument_cites_specific_dollar_amounts
        ),
        "feat_argument_cites_contract_or_document": vector.argument_cites_contract_or_document,
        "feat_argument_has_chronological_timeline": vector.argument_has_chronological_timeline,
        "feat_argument_names_specific_witnesses": vector.argument_names_specific_witnesses,
        "feat_argument_quantifies_each_damage_component": (
            vector.argument_quantifies_each_damage_component
        ),
        "feat_argument_cites_statute_or_legal_basis": (
            vector.argument_cites_statute_or_legal_basis
        ),
        "feat_argument_identifies_specific_location": vector.argument_identifies_specific_location,
        "feat_claim_amount_stated_in_dollars": vector.claim_amount_stated_in_dollars,
        "feat_plaintiff_count": vector.plaintiff_count,
        "feat_defendant_count": vector.defendant_count,
        "feat_witness_count": vector.witness_count,
        "feat_text_length": vector.text_length,
        "feat_document_count": vector.document_count,
        "feat_attempted_mediation": vector.attempted_mediation,
        "feat_claim_amount_is_within_small_claims_limit": vector.claim_amount_is_within_small_claims_limit,
        "feat_damages_are_ongoing": vector.damages_are_ongoing,
        "feat_damages_have_third_party_valuation": vector.damages_have_third_party_valuation,
        "feat_damages_include_lost_wages": vector.damages_include_lost_wages,
        "feat_damages_include_out_of_pocket_costs": vector.damages_include_out_of_pocket_costs,
        "feat_damages_include_property_value_loss": vector.damages_include_property_value_loss,
        "feat_gave_opportunity_to_cure": vector.gave_opportunity_to_cure,
        "feat_has_signed_contract_attached": vector.has_signed_contract_attached,
        "feat_opposing_party_filed_response_documents": vector.opposing_party_filed_response_documents,
        "feat_sent_certified_mail": vector.sent_certified_mail,
        "feat_sent_written_demand_letter": vector.sent_written_demand_letter,
        "feat_user_seeks_court_costs": vector.user_seeks_court_costs,
        "feat_user_seeks_interest": vector.user_seeks_interest,
        "feat_contract_present": vector.contract_present,
        "feat_opposing_party_has_attorney": vector.opposing_party_has_attorney,
    }


def feature_vector_to_model_frame(vector: FeatureVector) -> pd.DataFrame:
    """Convert one API feature vector to a validated model matrix row."""
    raw = pd.DataFrame([feature_vector_to_raw_row(vector)])
    missing_values = [col for col in RAW_MODEL_FEATURE_COLUMNS if raw[col].isna().any()]
    if missing_values:
        raise ValueError(f"Missing required inference features: {missing_values}")

    x, _ = preprocess_feature_frame(raw, drop_missing=False)
    return x
