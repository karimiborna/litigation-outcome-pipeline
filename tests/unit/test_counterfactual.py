"""Tests for counterfactual analysis."""

import numpy as np
import pytest

from counterfactual.analyzer import (
    CounterfactualAnalyzer,
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
