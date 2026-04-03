# API Module

Inference service for serving predictions, retrievals, and counterfactual analyses.

## Responsibilities

- Expose REST endpoints for:
  - **`GET /`** — welcome JSON (assignment-style root route)
  - **`GET /health`** — `models_loaded`, `classifier_loaded`, `regressor_loaded` (registry load status)
  - **`POST /predict`** — Pydantic-validated body → LLM feature extraction → sklearn classifier + regressor → win probability + expected monetary outcome
  - **Similar cases** — return top-K similar historical cases with explanations
  - **Counterfactual** — accept feature changes, return outcome deltas
- Load **Production**-stage sklearn models from the MLflow model registry (`models.tracking.load_production_model`, `models:/<name>/Production`)
- Run the feature extraction pipeline on incoming raw case data
- Handle request validation (Pydantic on `/predict` before inference), error responses, and health checks

## Key Considerations

- Separate services for feature extraction (LLM calls) and model inference (fast)
- Feature extraction has higher latency due to LLM calls — consider async or pre-extraction
- Models are loaded from MLflow — service needs **reachable** `MLFLOW_TRACKING_URI` (and registry access for artifact download)
- Sklearn `predict` / `predict_proba` run in **`asyncio.to_thread`** so async routes do not block the event loop on CPU work
- Must support both batch and real-time inference
- Input validation is critical — this is the system boundary
- Containerized via Docker (see `docker/Dockerfile.inference` and root README)
- Deployed to AWS ECS or Google Cloud Run (see `infra/`)
