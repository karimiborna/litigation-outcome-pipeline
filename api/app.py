"""FastAPI application with prediction, retrieval, and counterfactual endpoints."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.dependencies import app_state
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CounterfactualItem,
    CounterfactualRequest,
    CounterfactualResponse,
    HealthResponse,
    LexRatioAnalysisRequest,
    LexRatioAnalysisResponse,
    LexRatioSignals,
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

# Add CORS middleware to allow frontend requests from everywhere (development friendly)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:5173",
        "https://*.vercel.app",
        "*",  # Allow all origins in development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for LexRatio frontend
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    logger.info("Static files mounted from %s", static_dir)


@app.get("/api/", response_model=RootResponse)
async def api_root() -> RootResponse:
    """API root endpoint — returns JSON."""
    return RootResponse(
        message="Welcome to the Litigation Outcome Predictor API.",
        service="litigation-outcome-pipeline",
        docs="/docs",
    )


@app.get("/")
async def root():
    """Serve the LexRatio UI as the homepage."""
    from fastapi.responses import FileResponse

    # Try root directory first, then static
    root_dir = Path(__file__).parent.parent
    lexratio_file = root_dir / "lexratio.html"
    if not lexratio_file.exists():
        static_dir = Path(__file__).parent / "static"
        lexratio_file = static_dir / "lexratio.html"

    if lexratio_file.exists():
        return FileResponse(lexratio_file, media_type="text/html")

    # Fallback to JSON if HTML not found
    return RootResponse(
        message="Welcome to the Litigation Outcome Predictor API.",
        service="litigation-outcome-pipeline",
        docs="/docs",
    )


@app.get("/lexratio")
async def lexratio_ui():
    """Serve the LexRatio UI (also available at root path)."""
    from fastapi.responses import FileResponse

    root_dir = Path(__file__).parent.parent
    lexratio_file = root_dir / "lexratio.html"
    if not lexratio_file.exists():
        static_dir = Path(__file__).parent / "static"
        lexratio_file = static_dir / "lexratio.html"
    if not lexratio_file.exists():
        raise HTTPException(status_code=404, detail="LexRatio UI not available")
    return FileResponse(lexratio_file, media_type="text/html")


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


@app.post("/api/analyze-lexratio", response_model=LexRatioAnalysisResponse)
async def analyze_lexratio(request: LexRatioAnalysisRequest) -> LexRatioAnalysisResponse:
    """Analyze a small claims case for LexRatio frontend."""
    if not app_state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if app_state.feature_extractor is None:
        raise HTTPException(status_code=503, detail="Feature extractor not initialized")

    # Build a case from the request
    case = ProcessedCase(
        case_number="LEXRATIO_UNKNOWN",
        case_title=request.case_title or "Small Claims Case",
        cause_of_action=request.cause_of_action or "other",
        filing_date=date.today(),
        full_text=request.case_text,
        claim_amount=request.claim_amount,
    )

    # Extract features and run prediction
    feature_vector = await app_state.feature_extractor.extract(case)
    prediction = await asyncio.to_thread(_run_prediction_sync, feature_vector, None)

    # Analyze case text for signals
    text_lower = request.case_text.lower()
    signals = LexRatioSignals(
        has_written_evidence=_detect_written_evidence(text_lower),
        sent_demand_letter=_detect_demand_letter(text_lower),
        has_contract=_detect_contract(text_lower),
        defendant_responded=_detect_defendant_response(text_lower),
        has_witnesses=_detect_witnesses(text_lower),
        damages_itemized=_detect_itemized_damages(text_lower),
    )

    # Generate strengths and weaknesses based on signals
    strengths = _generate_strengths(signals, prediction.win_probability)
    weaknesses = _generate_weaknesses(signals, prediction.win_probability)

    # Generate verdict summary
    verdict = _generate_verdict_summary(
        prediction.win_probability, prediction.expected_monetary_outcome
    )

    # Generate advice
    advice = _generate_advice(signals, prediction.win_probability, strengths, weaknesses)

    return LexRatioAnalysisResponse(
        win_probability=int(round(prediction.win_probability * 100)),
        expected_award=prediction.expected_monetary_outcome,
        confidence=prediction.confidence,
        verdict_summary=verdict,
        strengths=strengths,
        weaknesses=weaknesses,
        advice=advice,
        signals=signals,
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


def _detect_written_evidence(text: str) -> bool | None:
    """Check for mentions of written evidence."""
    keywords = [
        "contract",
        "receipt",
        "email",
        "text message",
        "invoice",
        "letter",
        "document",
        "agreement",
    ]
    return any(kw in text for kw in keywords) if text else None


def _detect_demand_letter(text: str) -> bool | None:
    """Check for demand letter or notice."""
    keywords = ["demand letter", "sent demand", "certified mail", "notice", "demand for payment"]
    return any(kw in text for kw in keywords) if text else None


def _detect_contract(text: str) -> bool | None:
    """Check for contract mention."""
    keywords = ["contract", "agreement", "terms", "signed"]
    return any(kw in text for kw in keywords) if text else None


def _detect_defendant_response(text: str) -> bool | None:
    """Check for defendant response."""
    keywords = [
        "defendant respond",
        "defendant said",
        "they admitted",
        "they denied",
        "defendant claim",
        "in response",
    ]
    return any(kw in text for kw in keywords) if text else None


def _detect_witnesses(text: str) -> bool | None:
    """Check for witnesses."""
    keywords = ["witness", "witnessed", "saw", "observed", "present", "testif"]
    return any(kw in text for kw in keywords) if text else None


def _detect_itemized_damages(text: str) -> bool | None:
    """Check for itemized damages."""
    keywords = ["itemized", "$", "amount", "cost", "repair", "replacement"]
    return any(kw in text for kw in keywords) if text else None


def _generate_strengths(signals: LexRatioSignals, win_prob: float) -> list[str]:
    """Generate case strengths based on signals."""
    strengths = []

    if signals.has_written_evidence:
        strengths.append("Documentary evidence supports claim")

    if signals.has_contract:
        strengths.append("Written contract establishes obligations")

    if signals.sent_demand_letter:
        strengths.append("Proper notice provided via demand letter")

    if signals.has_witnesses:
        strengths.append("Corroborating witness testimony available")

    if signals.damages_itemized:
        strengths.append("Damages clearly itemized and documented")

    if not strengths and win_prob > 0.5:
        strengths.append("Reasonable claim with supportable facts")

    return strengths[:4]  # Return up to 4


def _generate_weaknesses(signals: LexRatioSignals, win_prob: float) -> list[str]:
    """Generate case weaknesses based on signals."""
    weaknesses = []

    if not signals.has_written_evidence:
        weaknesses.append("Limited documentary evidence")

    if not signals.has_contract:
        weaknesses.append("No written agreement to reference")

    if not signals.sent_demand_letter:
        weaknesses.append("No formal demand made before filing")

    if not signals.defendant_responded:
        weaknesses.append("Lack of defendant's position on record")

    if win_prob < 0.5 and not signals.damages_itemized:
        weaknesses.append("Damages not clearly quantified")

    return weaknesses[:4]  # Return up to 4


def _generate_verdict_summary(win_prob: float, expected_award: float) -> str:
    """Generate a plain-English summary of the verdict."""
    if win_prob >= 0.75:
        outcome = "likely to prevail"
    elif win_prob >= 0.6:
        outcome = "moderately likely to prevail"
    elif win_prob >= 0.4:
        outcome = "could prevail with strong presentation"
    else:
        outcome = "faces significant challenges"

    award_str = f"with an expected recovery of ${int(expected_award)}" if expected_award > 0 else ""

    return f"Claimant is {outcome} {award_str}.".strip()


def _generate_advice(
    signals: LexRatioSignals, win_prob: float, strengths: list[str], weaknesses: list[str]
) -> str:
    """Generate counsel advice."""
    advice_parts = []

    if win_prob >= 0.65:
        advice_parts.append(
            "Case presents favorable prospects. Focus on presenting"
            "evidence clearly and concisely at hearing."
        )
    elif win_prob >= 0.4:
        advice_parts.append(
            "Case has merit but requires strong presentation. Organize evidence"
            "logically and address potential counterarguments."
        )
    else:
        advice_parts.append(
            "Consider settlement discussions given case weaknesses. If proceeding,"
            "emphasize strongest points and address gaps in evidence."
        )

    if not signals.sent_demand_letter:
        advice_parts.append(
            "Consider whether prior demand letter would have strengthened negotiating position."
        )

    if not signals.damages_itemized:
        advice_parts.append(
            "Clearly itemize all damages with supporting receipts and documentation."
        )

    return " ".join(advice_parts)
