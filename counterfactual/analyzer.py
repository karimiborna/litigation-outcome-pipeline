"""Counterfactual analysis — simulates feature changes and shows predicted outcome shifts."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from features.schema import FeatureVector

logger = logging.getLogger(__name__)

FEATURE_CONSTRAINTS: dict[str, dict[str, Any]] = {
    "evidence_strength": {"min": 1.0, "max": 5.0, "type": "int"},
    "argument_clarity_plaintiff": {"min": 1.0, "max": 5.0, "type": "int"},
    "argument_clarity_defendant": {"min": 1.0, "max": 5.0, "type": "int"},
    "timeline_clarity": {"min": 1.0, "max": 5.0, "type": "int"},
    "monetary_amount_claimed": {"min": 0.0, "max": None, "type": "float"},
    "witness_count": {"min": 0.0, "max": None, "type": "int"},
    "contract_present": {"min": 0.0, "max": 1.0, "type": "bool"},
    "documentary_evidence": {"min": 0.0, "max": 1.0, "type": "bool"},
    "prior_attempts_to_resolve": {"min": 0.0, "max": 1.0, "type": "bool"},
    "legal_representation_plaintiff": {"min": 0.0, "max": 1.0, "type": "bool"},
    "legal_representation_defendant": {"min": 0.0, "max": 1.0, "type": "bool"},
    "counterclaim_present": {"min": 0.0, "max": 1.0, "type": "bool"},
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
        base_input = feature_vector.to_model_input()
        base_df = pd.DataFrame([base_input])

        base_win_prob = float(self._classifier.predict_proba(base_df)[0, 1])
        base_monetary = float(self._regressor.predict(base_df)[0])

        if perturbations is None:
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
