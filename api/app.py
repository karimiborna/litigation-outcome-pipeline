"""FastAPI application with prediction, retrieval, and counterfactual endpoints."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, HTTPException

from api.dependencies import app_state
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CounterfactualItem,
    CounterfactualRequest,
    CounterfactualResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
    RootResponse,
    SimilarCaseItem,
    SimilarCaseRequest,
    SimilarCaseResponse,
)
from data.schemas.case import ProcessedCase
from features.schema import FeatureVector

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and services on startup."""
    logger.info("Starting up — loading models and services...")
    app_state.load_feature_extractor()
    app_state.load_models()
    app_state.load_case_index()
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Litigation Outcome Predictor",
    description="Predicts small claims court case outcomes using ML",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    return RootResponse(
        message="Welcome to the Litigation Outcome Predictor API.",
        service="litigation-outcome-pipeline",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        models_loaded=app_state.models_loaded,
        classifier_loaded=app_state.classifier_loaded,
        regressor_loaded=app_state.regressor_loaded,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict win probability and expected monetary outcome for a case."""
    if not app_state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if app_state.feature_extractor is None:
        raise HTTPException(status_code=503, detail="Feature extractor not initialized")

    case = _build_processed_case(request)
    feature_vector = await app_state.feature_extractor.extract(case)

    return await asyncio.to_thread(_run_prediction_sync, feature_vector, request.case_number)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Batch prediction for multiple cases."""
    if not app_state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if app_state.feature_extractor is None:
        raise HTTPException(status_code=503, detail="Feature extractor not initialized")

    predictions: list[PredictionResponse] = []
    for case_req in request.cases:
        case = _build_processed_case(case_req)
        vector = await app_state.feature_extractor.extract(case)
        pred = await asyncio.to_thread(_run_prediction_sync, vector, case_req.case_number)
        predictions.append(pred)

    return BatchPredictionResponse(predictions=predictions, total=len(predictions))


@app.post("/similar", response_model=SimilarCaseResponse)
async def similar_cases(request: SimilarCaseRequest) -> SimilarCaseResponse:
    """Find similar historical cases."""
    if app_state.case_index is None:
        raise HTTPException(status_code=503, detail="Case index not available")

    results = app_state.case_index.search(request.case_text, top_k=request.top_k)

    items = [
        SimilarCaseItem(
            case_number=r.case_number,
            case_title=r.case_title,
            similarity_score=r.score,
            outcome=r.metadata.get("outcome"),
        )
        for r in results
    ]

    if items:
        outcomes = [i.outcome for i in items if i.outcome]
        explanation = f"Found {len(items)} similar cases."
        if outcomes:
            explanation += f" Most common outcome: {max(set(outcomes), key=outcomes.count)}."
    else:
        explanation = "No similar cases found above the similarity threshold."

    return SimilarCaseResponse(
        query_summary=request.case_text[:200],
        similar_cases=items,
        explanation=explanation,
    )


@app.post("/counterfactual", response_model=CounterfactualResponse)
async def counterfactual(request: CounterfactualRequest) -> CounterfactualResponse:
    """Analyze how feature changes would affect the predicted outcome."""
    if not app_state.models_loaded or app_state.counterfactual_analyzer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if app_state.feature_extractor is None:
        raise HTTPException(status_code=503, detail="Feature extractor not initialized")

    case = _build_processed_case_from_cf(request)
    feature_vector = await app_state.feature_extractor.extract(case)

    results = app_state.counterfactual_analyzer.analyze(
        feature_vector, perturbations=request.perturbations
    )

    items = [
        CounterfactualItem(
            feature=r.feature_name,
            original_value=r.original_value,
            new_value=r.new_value,
            win_probability_delta=round(r.win_prob_delta, 4),
            monetary_outcome_delta=round(r.monetary_delta, 2),
            description=r._describe(),
        )
        for r in results
    ]

    original_win = results[0].original_win_prob if results else 0.0
    original_monetary = results[0].original_monetary if results else 0.0

    return CounterfactualResponse(
        case_number=request.case_number,
        original_win_probability=round(original_win, 4),
        original_monetary_outcome=round(original_monetary, 2),
        counterfactuals=items,
    )


def _build_processed_case(request: PredictionRequest) -> ProcessedCase:
    return ProcessedCase(
        case_number=request.case_number or "UNKNOWN",
        case_title=request.case_title or "",
        cause_of_action=request.cause_of_action,
        filing_date=(
            date.fromisoformat(request.filing_date) if request.filing_date else date.today()
        ),
        full_text=request.case_text,
        claim_amount=request.claim_amount,
    )


def _build_processed_case_from_cf(request: CounterfactualRequest) -> ProcessedCase:
    return ProcessedCase(
        case_number=request.case_number or "UNKNOWN",
        case_title=request.case_title or "",
        cause_of_action=request.cause_of_action,
        filing_date=(
            date.fromisoformat(request.filing_date) if request.filing_date else date.today()
        ),
        full_text=request.case_text,
    )


def _run_prediction_sync(vector: FeatureVector, case_number: str | None) -> PredictionResponse:
    import pandas as pd

    model_input = pd.DataFrame([vector.to_model_input()])
    win_prob = float(app_state.classifier.predict_proba(model_input)[0, 1])
    monetary = float(app_state.regressor.predict(model_input)[0])

    if win_prob > 0.7:
        confidence = "high"
    elif win_prob > 0.4:
        confidence = "medium"
    else:
        confidence = "low"

    return PredictionResponse(
        case_number=case_number,
        win_probability=round(win_prob, 4),
        expected_monetary_outcome=round(monetary, 2),
        confidence=confidence,
    )
