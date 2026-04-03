"""Tests for API request/response schemas."""

import pytest
from pydantic import ValidationError

from api.schemas import (
    BatchPredictionRequest,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RootResponse,
    SimilarCaseRequest,
)


class TestPredictionRequest:
    def test_valid(self):
        r = PredictionRequest(case_text="This is a small claims case about unpaid rent.")
        assert r.case_number is None
        assert r.claim_amount is None

    def test_text_too_short(self):
        with pytest.raises(ValidationError):
            PredictionRequest(case_text="Short")

    def test_full_request(self):
        r = PredictionRequest(
            case_text="The plaintiff alleges the defendant owes $2000 for services rendered.",
            case_number="SC26001",
            case_title="DOE vs SMITH",
            cause_of_action="SMALL CLAIMS",
            filing_date="2026-01-15",
            claim_amount=2000.0,
        )
        assert r.claim_amount == 2000.0


class TestPredictionResponse:
    def test_create(self):
        r = PredictionResponse(
            win_probability=0.75,
            expected_monetary_outcome=1500.0,
            confidence="high",
        )
        assert r.win_probability == 0.75


class TestSimilarCaseRequest:
    def test_default_top_k(self):
        r = SimilarCaseRequest(case_text="A dispute over property damage.")
        assert r.top_k == 5

    def test_custom_top_k(self):
        r = SimilarCaseRequest(case_text="A dispute over property damage.", top_k=10)
        assert r.top_k == 10

    def test_top_k_bounds(self):
        with pytest.raises(ValidationError):
            SimilarCaseRequest(case_text="Some case text here.", top_k=0)
        with pytest.raises(ValidationError):
            SimilarCaseRequest(case_text="Some case text here.", top_k=50)


class TestBatchPredictionRequest:
    def test_valid(self):
        r = BatchPredictionRequest(
            cases=[PredictionRequest(case_text="Case about unpaid debt of $1000.")]
        )
        assert len(r.cases) == 1


class TestHealthResponse:
    def test_create(self):
        r = HealthResponse(
            status="healthy",
            version="0.1.0",
            models_loaded=True,
            classifier_loaded=True,
            regressor_loaded=True,
        )
        assert r.models_loaded is True
        assert r.classifier_loaded is True
        assert r.regressor_loaded is True


class TestRootResponse:
    def test_defaults(self):
        r = RootResponse(message="Hi", service="test")
        assert r.docs == "/docs"
