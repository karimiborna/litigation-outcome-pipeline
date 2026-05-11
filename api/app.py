"""FastAPI application with prediction, retrieval, and counterfactual endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.dependencies import app_state
from counterfactual.analyzer import (
    CounterfactualResult,
    format_for_llm,
    select_top_recommendations,
)
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
    RagAdviceEvaluation,
    RootResponse,
    SimilarCaseItem,
    SimilarCaseRequest,
    SimilarCaseResponse,
)
from data.schemas.case import ProcessedCase
from features.config import FeaturesConfig
from features.schema import FeatureVector
from models.dataset import feature_vector_to_model_frame
from api.prompts import build_rag_advice_judge_prompt, build_similarity_advice_prompt

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

# Add CORS middleware to allow frontend requests from Vercel and localhost
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


def _lexratio_file() -> Path:
    static_file = Path(__file__).parent / "static" / "lexratio.html"
    if static_file.exists():
        return static_file
    return Path(__file__).parent.parent / "lexratio.html"


@app.get("/api/", response_model=RootResponse)
async def api_root() -> RootResponse:
    return RootResponse(
        message="Welcome to the Litigation Outcome Predictor API.",
        service="litigation-outcome-pipeline",
    )


@app.get("/lexratio")
@app.get("/", include_in_schema=False)
async def lexratio_ui() -> FileResponse:
    """Serve the LexRatio UI."""
    lexratio_file = _lexratio_file()
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


@app.post("/analyze", response_model=LexRatioAnalysisResponse)
async def analyze_case(request: LexRatioAnalysisRequest) -> LexRatioAnalysisResponse:
    """Frontend-friendly analysis endpoint for lexratio.html."""
    if not app_state.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if app_state.feature_extractor is None:
        raise HTTPException(status_code=503, detail="Feature extractor not initialized")

    case_text = _build_frontend_case_text(request)
    case = ProcessedCase(
        case_number="LEXRATIO",
        case_title=request.case_title or "LexRatio frontend submission",
        cause_of_action=request.cause_of_action or "other",
        filing_date=date.today(),
        full_text=case_text,
        claim_amount=request.claim_amount,
    )
    feature_vector = await app_state.feature_extractor.extract(case)
    prediction = await asyncio.to_thread(_run_prediction_sync, feature_vector, "LEXRATIO")
    rag_context = await _build_rag_context(case_text, feature_vector=feature_vector)

    return _build_frontend_analysis_response(request, feature_vector, prediction, rag_context)


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


def _outcome_rank(outcome: str | None) -> int:
    ranking = {
        "plaintiff_win": 4,
        "settled": 3,
        "settlement": 3,
        "dismissed": 2,
        "defendant_win": 1,
    }
    return ranking.get(outcome, 0)


def _select_best_similar_cases(
    items: list[SimilarCaseItem], max_best: int = 3
) -> list[SimilarCaseItem]:
    return sorted(
        items,
        key=lambda item: (_outcome_rank(item.outcome), item.similarity_score),
        reverse=True,
    )[:max_best]


async def _build_similarity_advice(
    case_text: str,
    best_cases: list[SimilarCaseItem],
    perturbation_summary: str | None = None,
) -> tuple[str, str]:
    config = FeaturesConfig()
    if not best_cases:
        return (
            "No strong historical cases were found for comparison.",
            "Review evidence strength and documentation, and consider whether additional proof would make the claim clearer.",
        )

    if not config.llm_api_key:
        return _fallback_similarity_advice(best_cases)

    retrieved_cases = _similar_cases_for_prompt(best_cases)
    messages = build_similarity_advice_prompt(
        case_text, retrieved_cases, perturbation_summary=perturbation_summary
    )
    body = {
        "model": config.llm_model,
        "messages": messages,
        "temperature": config.llm_temperature,
        "max_tokens": config.llm_max_tokens,
    }
    base_url = config.llm_base_url or "https://api.openai.com/v1"
    headers = {
        "Authorization": f"Bearer {config.llm_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=config.llm_timeout) as client:
            resp = await client.post(f"{base_url}/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

        parsed = _parse_json_response(content)
        insights = parsed.get("comparison_insights")
        advice = parsed.get("advice")
        if not isinstance(insights, str) or not isinstance(advice, str):
            logger.warning(
                "LLM returned non-string advice fields (insights=%s, advice=%s); using fallback",
                type(insights).__name__,
                type(advice).__name__,
            )
            return _fallback_similarity_advice(best_cases)
        return insights, advice
    except Exception:
        logger.exception("RAG advice generation failed; using fallback advice")
        return _fallback_similarity_advice(best_cases)


async def _evaluate_rag_advice(
    case_text: str,
    best_cases: list[SimilarCaseItem],
    comparison_insights: str,
    advice: str,
    perturbation_summary: str | None = None,
) -> RagAdviceEvaluation:
    config = FeaturesConfig()
    if not best_cases:
        return RagAdviceEvaluation(
            score=3,
            verdict="needs_review",
            rationale="No retrieved cases were available, so the advice could not be grounded.",
        )

    if not config.llm_api_key:
        return _fallback_rag_advice_evaluation(advice, best_cases)

    retrieved_cases = _similar_cases_for_prompt(best_cases)
    messages = build_rag_advice_judge_prompt(
        case_text=case_text,
        retrieved_cases=retrieved_cases,
        advice=advice,
        comparison_insights=comparison_insights,
        perturbation_summary=perturbation_summary,
    )
    body = {
        "model": config.llm_model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512,
    }
    base_url = config.llm_base_url or "https://api.openai.com/v1"
    headers = {
        "Authorization": f"Bearer {config.llm_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=config.llm_timeout) as client:
            resp = await client.post(f"{base_url}/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

        return RagAdviceEvaluation.model_validate(_parse_json_response(content))
    except Exception:
        logger.exception("RAG advice evaluation failed; using fallback evaluation")
        return _fallback_rag_advice_evaluation(advice, best_cases)


def _parse_json_response(content: str) -> dict:
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _similar_cases_for_prompt(best_cases: list[SimilarCaseItem]) -> list[dict]:
    return [
        {
            "case_number": getattr(case, "case_number", "unknown"),
            "case_title": getattr(case, "case_title", "unknown"),
            "outcome": getattr(case, "outcome", "unknown"),
            "similarity_score": getattr(case, "similarity_score", 0.0),
            "case_snippet": getattr(case, "case_snippet", ""),
        }
        for case in best_cases
    ]

def _fallback_similarity_advice(best_cases: list[SimilarCaseItem]) -> tuple[str, str]:
    winner = best_cases[0]
    outcome = winner.outcome or "unknown"
    summary = (
        f"The strongest retrieved comparison is {winner.case_number} ({winner.case_title}) "
        f"with outcome {outcome}."
    )
    advice = (
        "The historical cases that did best tend to have strong evidence, clear documentation, "
        "and a well-articulated timeline. Focus on making your claim concrete and tied to specific facts."
    )
    return summary, advice


def _fallback_rag_advice_evaluation(
    advice: str, best_cases: list[SimilarCaseItem]
) -> RagAdviceEvaluation:
    has_grounding = bool(best_cases)
    has_practical_terms = any(
        term in advice.lower()
        for term in ("evidence", "document", "timeline", "receipt", "contract", "fact")
    )
    score = 4 if has_grounding and has_practical_terms else 3
    return RagAdviceEvaluation(
        score=score,
        verdict="pass" if score >= 4 else "needs_review",
        rationale=(
            "Fallback judge found the advice practical and connected to retrieved cases."
            if score >= 4
            else "Fallback judge could not confirm strong grounding in retrieved cases."
        ),
    )


async def _build_perturbation_context(
    feature_vector: FeatureVector | None,
) -> tuple[str | None, list[CounterfactualResult]]:
    """Run counterfactual analysis and produce both the LLM-readable summary
    and the structured top-5 results. Returns ``(None, [])`` when the analyzer
    is unavailable or fails."""
    if feature_vector is None or app_state.counterfactual_analyzer is None:
        return None, []
    try:
        all_results = await asyncio.to_thread(
            app_state.counterfactual_analyzer.analyze, feature_vector
        )
    except Exception:
        logger.exception("Counterfactual analysis failed; skipping perturbation summary")
        return None, []
    top_results = select_top_recommendations(all_results, top_n=5)
    return format_for_llm(top_results), top_results


async def _build_rag_context(
    case_text: str, feature_vector: FeatureVector | None = None
) -> dict:
    perturbation_summary, top_results = await _build_perturbation_context(feature_vector)

    if app_state.case_index is None:
        return {
            "similar_cases": [],
            "best_cases": [],
            "comparison_insights": "No retrieval index is available for comparison.",
            "advice": None,
            "advice_evaluation": None,
            "top_recommendations": top_results,
        }
    
    results = app_state.case_index.query(case_text, k=5)
    similar_cases = [
        SimilarCaseItem(
            case_number=r.document.id,
            case_title=r.document.metadata.get("case_title", r.document.id),
            outcome=r.document.metadata.get("outcome", "unknown"),
            similarity_score=getattr(r, "score", 0.0),
            case_snippet=r.document.metadata.get("case_snippet", "")
        )
        for r in results
    ]
    best_cases = _select_best_similar_cases(similar_cases)
    comparison_insights, advice = await _build_similarity_advice(
        case_text, best_cases, perturbation_summary=perturbation_summary
    )
    advice_evaluation = await _evaluate_rag_advice(
        case_text,
        best_cases,
        comparison_insights,
        advice,
        perturbation_summary=perturbation_summary,
    )
    return {
        "similar_cases": similar_cases,
        "best_cases": best_cases,
        "comparison_insights": comparison_insights,
        "advice": advice,
        "advice_evaluation": advice_evaluation,
        "top_recommendations": top_results,
    }


@app.post("/similar", response_model=SimilarCaseResponse)
async def similar_cases(request: SimilarCaseRequest) -> SimilarCaseResponse:
    """Find similar historical cases using the hybrid case index."""

    if app_state.case_index is None:
        raise HTTPException(status_code=503, detail="Case index not available")

    # Run the query through the hybrid index
    results = app_state.case_index.query(request.case_text, k=request.k)

    # Convert RetrievalResult -> SimilarCaseItem
    items = [
        SimilarCaseItem(
            case_number=r.document.id,
            case_title=r.document.metadata.get("case_title", r.document.id),
            similarity_score=r.score,
            outcome=r.document.metadata.get("outcome"),
            case_snippet=r.document.metadata.get("case_snippet") or r.document.text[:300],
        )
        for r in results
    ]

    best_cases = _select_best_similar_cases(items)

    # Build explanation
    if items:
        outcomes = [i.outcome for i in items if i.outcome]
        explanation = f"Found {len(items)} similar cases."
        if outcomes:
            most_common = max(set(outcomes), key=outcomes.count)
            explanation += f" Most common outcome: {most_common}."
    else:
        explanation = "No similar cases found above the similarity threshold."

    # Optional: build advice and evaluate RAG
    comparison_insights, advice = await _build_similarity_advice(request.case_text, best_cases)
    advice_evaluation = await _evaluate_rag_advice(
        request.case_text,
        best_cases,
        comparison_insights,
        advice,
    )

    return SimilarCaseResponse(
        query_summary=request.case_text[:200],
        similar_cases=items,
        best_cases=best_cases,
        explanation=explanation,
        comparison_insights=comparison_insights,
        advice=advice,
        advice_evaluation=advice_evaluation,
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

    try:
        results = app_state.counterfactual_analyzer.analyze(
            feature_vector, perturbations=request.perturbations
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # When the user did not pass custom perturbations, surface the curated top-5;
    # otherwise pass through whatever they asked about.
    surfaced = (
        select_top_recommendations(results, top_n=5)
        if request.perturbations is None
        else results
    )

    items = [
        CounterfactualItem(
            feature=r.feature_name,
            original_value=r.original_value,
            new_value=r.new_value,
            direction=r.direction,
            win_probability_delta=round(r.win_prob_delta, 4),
            monetary_outcome_delta=round(r.monetary_delta, 2),
            description=r.describe(),
        )
        for r in surfaced
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
    rag_context = await _build_rag_context(request.case_text, feature_vector=feature_vector)

    return _build_frontend_analysis_response(request, feature_vector, prediction, rag_context)


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
    try:
        model_input = feature_vector_to_model_frame(vector)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

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


def _build_frontend_case_text(request: LexRatioAnalysisRequest) -> str:
    parts = [request.case_text.strip()]
    return "\n\n".join(parts)


def _build_frontend_analysis_response(
    request: LexRatioAnalysisRequest,
    vector: FeatureVector,
    prediction: PredictionResponse,
    rag_context: dict | None = None,
) -> LexRatioAnalysisResponse:
    strengths: list[str] = []
    weaknesses: list[str] = []

    # Derived signals from v2 existence-based booleans
    documentary_evidence_flags = [
        vector.has_photos_or_physical_evidence,
        vector.has_receipts_or_financial_records,
        vector.has_written_communications,
        vector.has_invoices_or_billing_records,
        vector.has_signed_contract_attached,
    ]
    has_documentary_evidence: bool | None
    if any(f is True for f in documentary_evidence_flags):
        has_documentary_evidence = True
    elif all(f is False for f in documentary_evidence_flags):
        has_documentary_evidence = False
    else:
        has_documentary_evidence = None

    evidence_strength_count = sum(
        1
        for f in documentary_evidence_flags
        + [
            vector.has_witness_statements,
            vector.has_repair_or_replacement_estimate,
            vector.has_expert_assessment,
        ]
        if f is True
    )

    prior_resolution_flags = [
        vector.sent_written_demand_letter,
        vector.sent_certified_mail,
        vector.attempted_mediation,
        vector.gave_opportunity_to_cure,
    ]
    prior_attempts_to_resolve: bool | None
    if any(f is True for f in prior_resolution_flags):
        prior_attempts_to_resolve = True
    elif all(f is False for f in prior_resolution_flags):
        prior_attempts_to_resolve = False
    else:
        prior_attempts_to_resolve = None

    if has_documentary_evidence:
        strengths.append(
            "You appear to have documentary evidence that can support your version of events."
        )
    elif has_documentary_evidence is False:
        weaknesses.append(
            "The claim may be harder to"
            " prove without documents, receipts, photos, or written messages."
        )

    if vector.contract_present:
        strengths.append("A contract or agreement is a strong anchor for a small claims dispute.")
    elif vector.contract_present is False and request.cause_of_action in {
        "unpaid_debt",
        "service_dispute",
        "breach_of_contract",
    }:
        weaknesses.append(
            "This dispute may turn on oral promises unless you can show clear written terms."
        )

    if evidence_strength_count >= 4:
        strengths.append("The factual record appears comparatively strong and well-supported.")
    elif evidence_strength_count <= 1:
        weaknesses.append("The evidentiary record looks thin and may invite credibility disputes.")

    if prior_attempts_to_resolve:
        strengths.append("Prior efforts to resolve the dispute can help show reasonableness.")
    elif prior_attempts_to_resolve is False:
        weaknesses.append(
            "Bring proof that you tried to resolve the dispute before filing, if available."
        )

    if (vector.witness_count or 0) > 0:
        strengths.append("Witness support may reinforce your timeline and damages theory.")

    if (
        vector.argument_has_chronological_timeline is False
        or vector.argument_cites_specific_dates is False
    ):
        weaknesses.append("The timeline may need to be presented more clearly for the judge.")

    if vector.counterclaim_present:
        weaknesses.append(
            "There may be defenses or counterclaims that reduce recovery or complicate the hearing."
        )

    if not strengths:
        strengths.append(
            "The claim has enough stated detail to support a structured presentation at hearing."
        )
    if not weaknesses:
        weaknesses.append(
            "The main risk is whether the "
            "available proof fully matches the amount you are claiming."
        )

    strengths = strengths[:4]
    weaknesses = weaknesses[:4]

    win_probability = int(round(prediction.win_probability * 100))
    expected_award = max(0.0, prediction.expected_monetary_outcome)
    signals = LexRatioSignals(
        has_written_evidence=has_documentary_evidence,
        sent_demand_letter=vector.sent_written_demand_letter,
        has_contract=vector.contract_present,
        defendant_responded=vector.opposing_party_filed_response_documents,
        has_witnesses=(vector.witness_count > 0) if vector.witness_count is not None else None,
        damages_itemized=vector.argument_quantifies_each_damage_component,
    )

    claim_label = (request.cause_of_action or "small claims dispute").replace("_", " ")
    if win_probability >= 65:
        verdict_summary = (
            f"This {claim_label} appears more likely than not to succeed if the "
            "supporting proof is presented clearly."
        )
    elif win_probability >= 40:
        verdict_summary = (
            f"This {claim_label} looks contestable, with the outcome likely to "
            "depend on documentation and credibility."
        )
    else:
        verdict_summary = (
            f"This {claim_label} currently appears difficult to win without "
            "stronger proof or a clearer damages showing."
        )

    advice_parts = [
        "Organize your evidence in date order"
        " and tie each document to a specific fact you need the judge to accept.",
        "Prepare a short hearing narrative "
        "covering the agreement, the breach or harm, and the exact amount requested.",
    ]
    if prior_attempts_to_resolve is False:
        advice_parts.append(
            "If possible, bring a demand"
            " letter or other proof that you tried to resolve the dispute before hearing."
        )
    default_advice = " ".join(advice_parts[:3])
    rag_context = rag_context or {}
    advice = rag_context.get("advice") or default_advice

    top_results: list[CounterfactualResult] = rag_context.get("top_recommendations") or []
    top_recommendations = [
        CounterfactualItem(
            feature=r.feature_name,
            original_value=r.original_value,
            new_value=r.new_value,
            direction=r.direction,
            win_probability_delta=round(r.win_prob_delta, 4),
            monetary_outcome_delta=round(r.monetary_delta, 2),
            description=r.describe(),
        )
        for r in top_results
    ]

    return LexRatioAnalysisResponse(
        win_probability=win_probability,
        expected_award=round(expected_award, 2),
        confidence=prediction.confidence,
        verdict_summary=verdict_summary,
        strengths=strengths,
        weaknesses=weaknesses,
        advice=advice,
        signals=signals,
        similar_cases=rag_context.get("similar_cases") or [],
        best_cases=rag_context.get("best_cases") or [],
        comparison_insights=rag_context.get("comparison_insights"),
        advice_evaluation=rag_context.get("advice_evaluation"),
        top_recommendations=top_recommendations,
    )
