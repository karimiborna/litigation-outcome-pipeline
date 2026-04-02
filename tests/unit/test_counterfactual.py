"""Tests for counterfactual analysis."""

import numpy as np
import pytest

from counterfactual.analyzer import (
    PERTURBABLE_FEATURES,
    CounterfactualAnalyzer,
    CounterfactualResult,
)
from features.schema import FeatureVector


class FakeClassifier:
    """Fake classifier that returns higher win probability for stronger evidence."""

    def predict_proba(self, x):
        evidence = x.iloc[0]["evidence_strength"]
        prob = min(0.95, max(0.05, evidence / 5.0)) if evidence >= 0 else 0.5
        return np.array([[1 - prob, prob]])


class FakeRegressor:
    """Fake regressor that returns monetary amount proportional to claim."""

    def predict(self, x):
        claimed = x.iloc[0]["monetary_amount_claimed"]
        return np.array([claimed * 0.6 if claimed >= 0 else 1000.0])


@pytest.fixture
def analyzer():
    return CounterfactualAnalyzer(FakeClassifier(), FakeRegressor())


@pytest.fixture
def sample_vector():
    return FeatureVector(
        case_number="SC26001",
        evidence_strength=3,
        contract_present=True,
        argument_clarity_plaintiff=3,
        argument_clarity_defendant=2,
        monetary_amount_claimed=5000.0,
        witness_count=1,
        documentary_evidence=True,
        timeline_clarity=3,
        legal_representation_plaintiff=False,
        legal_representation_defendant=False,
        counterclaim_present=False,
        plaintiff_count=1,
        defendant_count=1,
        text_length=500,
        document_count=2,
    )


class TestCounterfactualAnalyzer:
    def test_explicit_perturbation(self, analyzer, sample_vector):
        results = analyzer.analyze(sample_vector, perturbations={"evidence_strength": 5.0})
        assert len(results) == 1
        assert results[0].feature_name == "evidence_strength"
        assert results[0].new_value == 5.0
        assert results[0].win_prob_delta > 0

    def test_auto_perturbations(self, analyzer, sample_vector):
        results = analyzer.analyze(sample_vector)
        assert len(results) > 0
        feature_names = [r.feature_name for r in results]
        assert any(f in PERTURBABLE_FEATURES for f in feature_names)

    def test_sorted_by_impact(self, analyzer, sample_vector):
        results = analyzer.analyze(sample_vector)
        deltas = [abs(r.win_prob_delta) for r in results]
        assert deltas == sorted(deltas, reverse=True)

    def test_result_to_dict(self):
        r = CounterfactualResult(
            feature_name="evidence_strength",
            original_value=3.0,
            new_value=5.0,
            original_win_prob=0.6,
            new_win_prob=0.8,
            original_monetary=3000.0,
            new_monetary=3000.0,
        )
        d = r.to_dict()
        assert d["feature"] == "evidence_strength"
        assert d["win_probability"]["delta"] == 0.2
        assert "increase" in d["description"]
