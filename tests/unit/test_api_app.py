"""Smoke tests for HTTP routes (no MLflow or LLM required)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import (
    _fallback_rag_advice_evaluation,
    _outcome_rank,
    _select_best_similar_cases,
    app,
)
from api.schemas import SimilarCaseItem


def test_get_root() -> None:
    with TestClient(app) as client:
        r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "message" in body
    assert body.get("docs") == "/docs"


def test_get_health() -> None:
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "healthy"
    assert "models_loaded" in body
    assert "classifier_loaded" in body
    assert "regressor_loaded" in body


def test_outcome_ranking() -> None:
    assert _outcome_rank("plaintiff_win") > _outcome_rank("settlement")
    assert _outcome_rank("defendant_win") > _outcome_rank(None)


def test_select_best_similar_cases() -> None:
    items = [
        SimilarCaseItem(
            case_number="A", case_title="A", similarity_score=0.5, outcome="settlement"
        ),
        SimilarCaseItem(
            case_number="B", case_title="B", similarity_score=0.8, outcome="plaintiff_win"
        ),
        SimilarCaseItem(
            case_number="C", case_title="C", similarity_score=0.9, outcome="defendant_win"
        ),
    ]
    best = _select_best_similar_cases(items)
    assert best[0].case_number == "B"


def test_fallback_rag_advice_evaluation() -> None:
    cases = [
        SimilarCaseItem(
            case_number="B",
            case_title="B",
            similarity_score=0.8,
            outcome="plaintiff_win",
        )
    ]

    result = _fallback_rag_advice_evaluation(
        "Use evidence, documents, and a clear timeline.", cases
    )

    assert result.score == 4
    assert result.verdict == "pass"
