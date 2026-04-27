"""Counterfactual analysis — simulates feature changes and shows predicted outcome shifts."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from features.schema import FeatureVector
from models.dataset import MODEL_FEATURE_COLUMNS, feature_vector_to_model_frame

logger = logging.getLogger(__name__)

FEATURE_CONSTRAINTS: dict[str, dict[str, Any]] = {
    # Numerics
    "monetary_amount_claimed": {"min": 0.0, "max": None, "type": "float"},
    "witness_count": {"min": 0.0, "max": None, "type": "int"},
    # Representation
    "user_has_attorney": {"min": 0.0, "max": 1.0, "type": "bool"},
    "opposing_party_has_attorney": {"min": 0.0, "max": 1.0, "type": "bool"},
    # Counter-filings / contract presence
    "counterclaim_present": {"min": 0.0, "max": 1.0, "type": "bool"},
    "contract_present": {"min": 0.0, "max": 1.0, "type": "bool"},
    # Evidence existence
    "has_photos_or_physical_evidence": {"min": 0.0, "max": 1.0, "type": "bool"},
    "has_receipts_or_financial_records": {"min": 0.0, "max": 1.0, "type": "bool"},
    "has_written_communications": {"min": 0.0, "max": 1.0, "type": "bool"},
    "has_witness_statements": {"min": 0.0, "max": 1.0, "type": "bool"},
    "has_signed_contract_attached": {"min": 0.0, "max": 1.0, "type": "bool"},
    "has_repair_or_replacement_estimate": {"min": 0.0, "max": 1.0, "type": "bool"},
    "has_police_report": {"min": 0.0, "max": 1.0, "type": "bool"},
    "has_medical_records": {"min": 0.0, "max": 1.0, "type": "bool"},
    "has_expert_assessment": {"min": 0.0, "max": 1.0, "type": "bool"},
    "has_invoices_or_billing_records": {"min": 0.0, "max": 1.0, "type": "bool"},
    # Argument content
    "argument_cites_specific_dates": {"min": 0.0, "max": 1.0, "type": "bool"},
    "argument_cites_specific_dollar_amounts": {"min": 0.0, "max": 1.0, "type": "bool"},
    "argument_cites_contract_or_document": {"min": 0.0, "max": 1.0, "type": "bool"},
    "argument_has_chronological_timeline": {"min": 0.0, "max": 1.0, "type": "bool"},
    "argument_names_specific_witnesses": {"min": 0.0, "max": 1.0, "type": "bool"},
    "argument_quantifies_each_damage_component": {"min": 0.0, "max": 1.0, "type": "bool"},
    "argument_cites_statute_or_legal_basis": {"min": 0.0, "max": 1.0, "type": "bool"},
    "argument_identifies_specific_location": {"min": 0.0, "max": 1.0, "type": "bool"},
    # Procedural / pre-filing conduct
    "sent_written_demand_letter": {"min": 0.0, "max": 1.0, "type": "bool"},
    "sent_certified_mail": {"min": 0.0, "max": 1.0, "type": "bool"},
    "gave_opportunity_to_cure": {"min": 0.0, "max": 1.0, "type": "bool"},
    "attempted_mediation": {"min": 0.0, "max": 1.0, "type": "bool"},
}

PERTURBABLE_FEATURES = list(FEATURE_CONSTRAINTS.keys())


class CounterfactualResult:
    """Result of a single counterfactual perturbation."""

    def __init__(
        self,
        feature_name: str,
        original_value: float,
        new_value: float,
        original_win_prob: float,
        new_win_prob: float,
        original_monetary: float,
        new_monetary: float,
    ):
        self.feature_name = feature_name
        self.original_value = original_value
        self.new_value = new_value
        self.original_win_prob = original_win_prob
        self.new_win_prob = new_win_prob
        self.original_monetary = original_monetary
        self.new_monetary = new_monetary

    @property
    def win_prob_delta(self) -> float:
        return self.new_win_prob - self.original_win_prob

    @property
    def monetary_delta(self) -> float:
        return self.new_monetary - self.original_monetary

    def to_dict(self) -> dict:
        return {
            "feature": self.feature_name,
            "original_value": self.original_value,
            "new_value": self.new_value,
            "win_probability": {
                "original": round(self.original_win_prob, 4),
                "new": round(self.new_win_prob, 4),
                "delta": round(self.win_prob_delta, 4),
            },
            "monetary_outcome": {
                "original": round(self.original_monetary, 2),
                "new": round(self.new_monetary, 2),
                "delta": round(self.monetary_delta, 2),
            },
            "description": self._describe(),
        }

    def _describe(self) -> str:
        direction = "increase" if self.win_prob_delta > 0 else "decrease"
        pct = abs(self.win_prob_delta) * 100
        return (
            f"If {self.feature_name} changed from {self.original_value} to "
            f"{self.new_value}, win probability would {direction} by {pct:.1f}%"
        )


class CounterfactualAnalyzer:
    """Analyzes how feature changes would affect predicted outcomes."""

    def __init__(self, classifier: Any, regressor: Any):
        self._classifier = classifier
        self._regressor = regressor

    def analyze(
        self,
        feature_vector: FeatureVector,
        perturbations: dict[str, float] | None = None,
    ) -> list[CounterfactualResult]:
        """Run counterfactual analysis on a single case.

        If perturbations is None, auto-generates meaningful perturbations
        for all perturbable features.
        """
        uses_v2_features = self._uses_v2_feature_space()
        if uses_v2_features:
            base_df = feature_vector_to_model_frame(feature_vector)
            expected = list(self._classifier.feature_names_in_)
            base_df = base_df.reindex(columns=expected)
            base_input = base_df.iloc[0].to_dict()
        else:
            base_input = feature_vector.to_model_input()
            base_df = pd.DataFrame([base_input])

        base_win_prob = float(self._classifier.predict_proba(base_df)[0, 1])
        base_monetary = float(self._regressor.predict(base_df)[0])

        if perturbations is None:
            if uses_v2_features:
                perturbations = self._auto_perturbations_v2(base_input)
            else:
                perturbations = self._auto_perturbations(base_input)

        results: list[CounterfactualResult] = []
        for feature_name, new_value in perturbations.items():
            if feature_name not in base_input:
                logger.warning("Unknown feature: %s", feature_name)
                continue

            new_value = self._clamp(feature_name, new_value)
            original_value = base_input[feature_name]

            if abs(new_value - original_value) < 1e-6:
                continue

            modified = base_input.copy()
            modified[feature_name] = new_value
            modified_df = pd.DataFrame([modified])
            if uses_v2_features:
                modified_df = modified_df.reindex(columns=list(self._classifier.feature_names_in_))

            new_win_prob = float(self._classifier.predict_proba(modified_df)[0, 1])
            new_monetary = float(self._regressor.predict(modified_df)[0])

            results.append(
                CounterfactualResult(
                    feature_name=feature_name,
                    original_value=original_value,
                    new_value=new_value,
                    original_win_prob=base_win_prob,
                    new_win_prob=new_win_prob,
                    original_monetary=base_monetary,
                    new_monetary=new_monetary,
                )
            )

        results.sort(key=lambda r: abs(r.win_prob_delta), reverse=True)
        return results

    def _uses_v2_feature_space(self) -> bool:
        expected = getattr(self._classifier, "feature_names_in_", None)
        if expected is None:
            return False
        return set(expected).issubset(set(MODEL_FEATURE_COLUMNS))

    def _auto_perturbations_v2(self, base_input: dict[str, float]) -> dict[str, float]:
        """Generate simple perturbations for the trained v2 feature frame."""
        perturbations: dict[str, float] = {}
        for feature_name, current in base_input.items():
            if feature_name.startswith("feat_claim_category_"):
                continue
            if feature_name in {"feat_text_length", "feat_document_count"}:
                continue
            if current in (0.0, 1.0):
                perturbations[feature_name] = 1.0 - current
            elif feature_name == "feat_monetary_amount_claimed":
                perturbations[feature_name] = current * 1.5 if current > 0 else 1000.0
            elif feature_name.endswith("_count") and current >= 0:
                perturbations[feature_name] = current + 1.0
        return perturbations

    def _auto_perturbations(self, base_input: dict[str, float]) -> dict[str, float]:
        """Generate meaningful perturbations for each perturbable feature."""
        perturbations: dict[str, float] = {}

        for feature_name in PERTURBABLE_FEATURES:
            if feature_name not in base_input:
                continue

            constraints = FEATURE_CONSTRAINTS[feature_name]
            current = base_input[feature_name]

            if current < 0:
                continue

            if constraints["type"] == "bool":
                perturbations[feature_name] = 1.0 - current
            elif constraints["type"] == "int":
                max_val = constraints.get("max")
                if max_val is not None and current < max_val:
                    perturbations[feature_name] = min(current + 1.0, max_val)
                elif constraints.get("min") is not None and current > constraints["min"]:
                    perturbations[feature_name] = max(current - 1.0, constraints["min"])
            elif constraints["type"] == "float":
                perturbations[feature_name] = current * 1.5 if current > 0 else 1000.0

        return perturbations

    def _clamp(self, feature_name: str, value: float) -> float:
        """Clamp a value to respect feature constraints."""
        if feature_name not in FEATURE_CONSTRAINTS:
            return value

        constraints = FEATURE_CONSTRAINTS[feature_name]
        min_val = constraints.get("min")
        max_val = constraints.get("max")

        if min_val is not None:
            value = max(value, min_val)
        if max_val is not None:
            value = min(value, max_val)
        return value
