"""Tests for counterfactual analysis (v2 existence-based feature schema)."""

import numpy as np
import pytest

from counterfactual.analyzer import (
    PERTURBABLE_FEATURES,
    CounterfactualAnalyzer,
    CounterfactualResult,
)
from features.schema import FeatureVector


class FakeClassifier:
    """Fake classifier: higher win probability when written communications exist."""

    def predict_proba(self, x):
        signal = x.iloc[0]["feat_has_written_communications"]
        prob = 0.8 if signal >= 1.0 else 0.3
        return np.array([[1 - prob, prob]])


class FakeRegressor:
    """Fake regressor: monetary outcome proportional to claim amount."""

    def predict(self, x):
        claimed = x.iloc[0]["feat_monetary_amount_claimed"]
        return np.array([claimed * 0.6 if claimed >= 0 else 1000.0])


@pytest.fixture
def analyzer():
    return CounterfactualAnalyzer(FakeClassifier(), FakeRegressor())


@pytest.fixture
def sample_vector():
    return FeatureVector(
        case_number="SC26001",
        user_is_plaintiff=True,
        contract_present=True,
        monetary_amount_claimed=5000.0,
        witness_count=1,
        user_has_attorney=False,
        opposing_party_has_attorney=False,
        has_written_communications=False,
        has_receipts_or_financial_records=True,
        argument_cites_specific_dates=True,
        argument_cites_specific_dollar_amounts=True,
        sent_written_demand_letter=False,
        counterclaim_present=False,
        plaintiff_count=1,
        defendant_count=1,
        text_length=500,
        document_count=2,
    )


class TestCounterfactualAnalyzer:
    def test_explicit_perturbation(self, analyzer, sample_vector):
        results = analyzer.analyze(sample_vector, perturbations={"feat_has_written_communications":
                                                                  1.0})
        assert len(results) == 1
        assert results[0].feature_name == "feat_has_written_communications"
        assert results[0].new_value == 1.0
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
            feature_name="feat_has_written_communications",
            original_value=0.0,
            new_value=1.0,
            original_win_prob=0.3,
            new_win_prob=0.8,
            original_monetary=3000.0,
            new_monetary=3000.0,
        )
        d = r.to_dict()
        assert d["feature"] == "feat_has_written_communications"
        assert d["win_probability"]["delta"] == 0.5
        assert "increase" in d["description"]
