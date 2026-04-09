"""Request and response schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    """Welcome payload for GET /."""

    message: str
    service: str
    docs: str = "/docs"


class PredictionRequest(BaseModel):
    case_text: str = Field(..., min_length=10)
    case_number: str | None = None
    case_title: str | None = None
    cause_of_action: str | None = None
    filing_date: str | None = None
    claim_amount: float | None = None


class PredictionResponse(BaseModel):
    case_number: str | None = None
    win_probability: float
    expected_monetary_outcome: float
    confidence: str


class SimilarCaseRequest(BaseModel):
    case_text: str = Field(..., min_length=10)
    top_k: int = Field(default=5, ge=1, le=20)


class SimilarCaseItem(BaseModel):
    case_number: str
    case_title: str
    similarity_score: float
    outcome: str | None = None


class SimilarCaseResponse(BaseModel):
    query_summary: str
    similar_cases: list[SimilarCaseItem]
    explanation: str


class CounterfactualRequest(BaseModel):
    case_text: str = Field(..., min_length=10)
    case_number: str | None = None
    case_title: str | None = None
    cause_of_action: str | None = None
    filing_date: str | None = None
    perturbations: dict[str, float] | None = None


class CounterfactualItem(BaseModel):
    feature: str
    original_value: float
    new_value: float
    win_probability_delta: float
    monetary_outcome_delta: float
    description: str


class CounterfactualResponse(BaseModel):
    case_number: str | None = None
    original_win_probability: float
    original_monetary_outcome: float
    counterfactuals: list[CounterfactualItem]


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool
    classifier_loaded: bool = False
    regressor_loaded: bool = False


class BatchPredictionRequest(BaseModel):
    cases: list[PredictionRequest] = Field(..., min_length=1, max_length=50)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    total: int
