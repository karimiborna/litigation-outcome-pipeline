"""Smoke tests for HTTP routes (no MLflow or LLM required)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import app


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
