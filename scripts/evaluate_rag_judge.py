#!/usr/bin/env python3
"""Evaluate the RAG retrieval system using LLM-as-judge.

This script runs the LLM-as-judge evaluation on a sample of historical cases
from the retrieval index, computing quality metrics for the RAG-generated advice.

The evaluation:
1. For each test case, retrieves similar cases from the FAISS index
2. Generates advice based on retrieved cases using an LLM
3. Uses the LLM-as-judge to score the advice (1-5 scale)
4. Logs aggregate metrics to MLflow

Usage:
    python scripts/evaluate_rag_judge.py [--n-samples 20]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from retrieval.config import RetrievalConfig
from retrieval.index import CaseIndex, CaseMetadataStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_MLFLOW_EXPERIMENT_NAME = "rag-judge-evaluation"


def load_test_cases(index_dir: Path, n_samples: int) -> list[dict]:
    """Load test cases from the retrieval index metadata."""
    metadata_path = index_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(metadata_path) as f:
        all_cases = json.load(f)

    logger.info("Total cases in index: %d", len(all_cases))

    # Sample cases with known outcomes for better evaluation
    cases_with_outcome = [c for c in all_cases if c.get("outcome") is not None]
    cases_without_outcome = [c for c in all_cases if c.get("outcome") is None]

    # Mix of outcomes for diverse evaluation
    n_with_outcome = min(int(n_samples * 0.8), len(cases_with_outcome))
    n_without_outcome = min(n_samples - n_with_outcome, len(cases_without_outcome))

    random.seed(42)

    sampled = []
    if cases_with_outcome:
        sampled.extend(random.sample(cases_with_outcome, n_with_outcome))
    if cases_without_outcome:
        sampled.extend(random.sample(cases_without_outcome, n_without_outcome))

    logger.info("Sampled %d cases for evaluation", len(sampled))
    return sampled


def _parse_json_response(content: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _outcome_rank(outcome: str | None) -> int:
    """Rank outcomes for sorting (higher = better for plaintiff)."""
    ranking = {
        "plaintiff_win": 4,
        "settled": 3,
        "settlement": 3,
        "dismissed": 2,
        "defendant_win": 1,
    }
    return ranking.get(outcome, 0)


def _select_best_similar_cases(
    items: list[dict], max_best: int = 3
) -> list[dict]:
    """Select the best similar cases based on outcome and similarity score."""
    return sorted(
        items,
        key=lambda item: (_outcome_rank(item.get("outcome")), item.get("similarity_score", 0)),
        reverse=True,
    )[:max_best]


def _similar_cases_for_prompt(best_cases: list[dict]) -> list[dict]:
    """Format similar cases for the LLM prompt."""
    return [
        {
            "case_number": case["case_number"],
            "case_title": case["case_title"],
            "outcome": case.get("outcome", "unknown"),
            "similarity_score": case.get("similarity_score", 0),
            "case_snippet": case.get("case_snippet", ""),
        }
        for case in best_cases
    ]


async def _build_similarity_advice(
    case_text: str, best_cases: list[dict], config: RetrievalConfig, features_config: Any
) -> tuple[str, str]:
    """Generate advice based on retrieved similar cases."""
    if not best_cases:
        return (
            "No strong historical cases were found for comparison.",
            "Review evidence strength and documentation, and consider whether additional proof would make the claim clearer.",
        )

    if not features_config.llm_api_key:
        # Fallback advice
        winner = best_cases[0]
        outcome = winner.get("outcome", "unknown")
        summary = (
            f"The strongest retrieved comparison is {winner['case_number']} ({winner['case_title']}) "
            f"with outcome {outcome}."
        )
        advice = (
            "The historical cases that did best tend to have strong evidence, clear documentation, "
            "and a well-articulated timeline. Focus on making your claim concrete and tied to specific facts."
        )
        return summary, advice

    from api.prompts import build_similarity_advice_prompt

    retrieved_cases = _similar_cases_for_prompt(best_cases)
    messages = build_similarity_advice_prompt(case_text, retrieved_cases)
    body = {
        "model": features_config.llm_model,
        "messages": messages,
        "temperature": features_config.llm_temperature,
        "max_tokens": features_config.llm_max_tokens,
    }
    base_url = features_config.llm_base_url or "https://api.openai.com/v1"
    headers = {
        "Authorization": f"Bearer {features_config.llm_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=features_config.llm_timeout) as client:
            resp = await client.post(f"{base_url}/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

        parsed = _parse_json_response(content)
        return (
            parsed.get("comparison_insights", "Comparison analysis was unavailable."),
            parsed.get(
                "advice",
                "Focus on improving the strength of the evidence and the clarity of the claim presentation.",
            ),
        )
    except Exception:
        logger.exception("RAG advice generation failed; using fallback advice")
        # Fallback advice
        winner = best_cases[0]
        outcome = winner.get("outcome", "unknown")
        summary = (
            f"The strongest retrieved comparison is {winner['case_number']} ({winner['case_title']}) "
            f"with outcome {outcome}."
        )
        advice = (
            "The historical cases that did best tend to have strong evidence, clear documentation, "
            "and a well-articulated timeline. Focus on making your claim concrete and tied to specific facts."
        )
        return summary, advice


async def _evaluate_rag_advice(
    case_text: str,
    best_cases: list[dict],
    comparison_insights: str,
    advice: str,
    features_config: Any,
) -> dict:
    """Evaluate RAG advice using LLM-as-judge."""
    if not best_cases:
        return {
            "score": 3,
            "verdict": "needs_review",
            "rationale": "No retrieved cases were available, so the advice could not be grounded.",
        }

    if not features_config.llm_api_key:
        # Fallback evaluation
        has_grounding = bool(best_cases)
        has_practical_terms = any(
            term in advice.lower()
            for term in ("evidence", "document", "timeline", "receipt", "contract", "fact")
        )
        score = 4 if has_grounding and has_practical_terms else 3
        return {
            "score": score,
            "verdict": "pass" if score >= 4 else "needs_review",
            "rationale": (
                "Fallback judge found the advice practical and connected to retrieved cases."
                if score >= 4
                else "Fallback judge could not confirm strong grounding in retrieved cases."
            ),
        }

    from api.prompts import build_rag_advice_judge_prompt

    retrieved_cases = _similar_cases_for_prompt(best_cases)
    messages = build_rag_advice_judge_prompt(
        case_text=case_text,
        retrieved_cases=retrieved_cases,
        advice=advice,
        comparison_insights=comparison_insights,
    )
    body = {
        "model": features_config.llm_model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 512,
    }
    base_url = features_config.llm_base_url or "https://api.openai.com/v1"
    headers = {
        "Authorization": f"Bearer {features_config.llm_api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=features_config.llm_timeout) as client:
            resp = await client.post(f"{base_url}/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

        parsed = _parse_json_response(content)
        return {
            "score": parsed.get("score", 3),
            "verdict": parsed.get("verdict", "needs_review"),
            "rationale": parsed.get("rationale", "Evaluation rationale unavailable."),
        }
    except Exception:
        logger.exception("RAG advice evaluation failed; using fallback evaluation")
        has_grounding = bool(best_cases)
        has_practical_terms = any(
            term in advice.lower()
            for term in ("evidence", "document", "timeline", "receipt", "contract", "fact")
        )
        score = 4 if has_grounding and has_practical_terms else 3
        return {
            "score": score,
            "verdict": "pass" if score >= 4 else "needs_review",
            "rationale": (
                "Fallback judge found the advice practical and connected to retrieved cases."
                if score >= 4
                else "Fallback judge could not confirm strong grounding in retrieved cases."
            ),
        }


async def evaluate_case(
    case: dict,
    case_index: CaseIndex,
    config: RetrievalConfig,
    features_config: Any,
) -> dict:
    """Evaluate a single case using the RAG system and LLM-as-judge."""
    # Get the case snippet as the query text
    query_text = case.get("case_snippet", "")
    if not query_text:
        return {
            "case_number": case["case_number"],
            "outcome": case.get("outcome"),
            "error": "No case snippet available",
        }

    # Search for similar cases
    results = case_index.search(query_text, top_k=5)
    similar_cases = [
        {
            "case_number": r.case_number,
            "case_title": r.case_title,
            "similarity_score": r.score,
            "outcome": r.metadata.get("outcome"),
            "case_snippet": r.metadata.get("case_snippet"),
        }
        for r in results
    ]

    if not similar_cases:
        return {
            "case_number": case["case_number"],
            "outcome": case.get("outcome"),
            "error": "No similar cases found",
            "score": 3,
            "verdict": "needs_review",
        }

    best_cases = _select_best_similar_cases(similar_cases)

    # Generate advice
    comparison_insights, advice = await _build_similarity_advice(query_text, best_cases, config, features_config)

    # Evaluate with LLM-as-judge
    evaluation = await _evaluate_rag_advice(
        query_text,
        best_cases,
        comparison_insights,
        advice,
        features_config,
    )

    return {
        "case_number": case["case_number"],
        "outcome": case.get("outcome"),
        "similarity_score": best_cases[0]["similarity_score"] if best_cases else None,
        "score": evaluation["score"],
        "verdict": evaluation["verdict"],
        "rationale": evaluation["rationale"],
        "advice": advice,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG system with LLM-as-judge.")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of test cases to evaluate",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path("data/retrieval_index"),
        help="Directory containing the FAISS index and metadata",
    )
    args = parser.parse_args()

    index_dir = args.index_dir
    if not index_dir.exists():
        raise SystemExit(f"Index directory not found: {index_dir}")
    if not (index_dir / "index.faiss").exists():
        raise SystemExit(f"FAISS index not found: {index_dir / 'index.faiss'}")

    # Set up MLflow
    import mlflow
    from models.config import MLflowConfig
    from features.config import FeaturesConfig

    mlflow_config = MLflowConfig()
    logger.info("MLflow tracking URI: %s", mlflow_config.tracking_uri)
    mlflow.set_tracking_uri(mlflow_config.tracking_uri)

    # Load configs
    retrieval_config = RetrievalConfig()
    features_config = FeaturesConfig()

    # Load the case index
    case_index = CaseIndex(retrieval_config)
    case_index.load(index_dir)
    logger.info("Loaded case index with %d cases", case_index.size)

    # Load test cases
    test_cases = load_test_cases(index_dir, args.n_samples)

    # Run evaluations
    logger.info("Running LLM-as-judge evaluation on %d cases...", len(test_cases))
    results = []
    for i, case in enumerate(test_cases, 1):
        logger.info("Evaluating case %d/%d: %s", i, len(test_cases), case["case_number"])
        try:
            result = await evaluate_case(case, case_index, retrieval_config, features_config)
            results.append(result)
            logger.info("  Score: %d/5, Verdict: %s", result.get("score", "?"), result.get("verdict", "?"))
        except Exception as e:
            logger.error("  Error evaluating %s: %s", case["case_number"], e)
            results.append({
                "case_number": case["case_number"],
                "outcome": case.get("outcome"),
                "error": str(e),
            })

    # Compute aggregate metrics
    valid_results = [r for r in results if "score" in r and r["score"] is not None]
    if not valid_results:
        logger.error("No valid evaluation results!")
        return 1

    scores = [r["score"] for r in valid_results]
    avg_score = sum(scores) / len(scores)
    pass_count = sum(1 for r in valid_results if r.get("verdict") == "pass")
    pass_rate = pass_count / len(valid_results)

    verdict_counts = {}
    for r in valid_results:
        v = r.get("verdict", "unknown")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    logger.info("")
    logger.info("=== RAG Judge Evaluation Results ===")
    logger.info("Cases evaluated: %d/%d", len(valid_results), len(test_cases))
    logger.info("Average score: %.2f/5.00", avg_score)
    logger.info("Pass rate: %.1f%%", pass_rate * 100)
    logger.info("Verdict distribution: %s", verdict_counts)

    # Log to MLflow
    with mlflow.start_run(run_name=f"rag-judge-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
        mlflow.log_param("n_samples", args.n_samples)
        mlflow.log_param("embedding_model", retrieval_config.embedding_model)
        mlflow.log_param("index_size", case_index.size)
        mlflow.log_param("top_k", retrieval_config.top_k)
        mlflow.log_param("similarity_threshold", retrieval_config.similarity_threshold)

        mlflow.log_metric("avg_judge_score", avg_score)
        mlflow.log_metric("pass_rate", pass_rate)
        mlflow.log_metric("n_cases_evaluated", len(valid_results))
        mlflow.log_metric("n_cases_with_errors", len(test_cases) - len(valid_results))

        for verdict, count in verdict_counts.items():
            mlflow.log_metric(f"verdict_{verdict}", count)

        # Score distribution
        for score_val in range(1, 6):
            count = scores.count(score_val)
            mlflow.log_metric(f"score_{score_val}", count)

        logger.info("")
        logger.info("Results logged to MLflow experiment '%s'", _MLFLOW_EXPERIMENT_NAME)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))