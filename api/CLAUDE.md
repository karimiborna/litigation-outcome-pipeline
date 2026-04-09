# API Module

FastAPI inference service — the user-facing layer of the pipeline.

## File Map

<<<<<<< HEAD
| File | Purpose |
|---|---|
| `app.py` | FastAPI app, lifespan, all endpoints |
| `dependencies.py` | AppState singleton — loads models/index on startup |
| `schemas.py` | Pydantic request/response models |
=======
- Expose REST endpoints for:
  - **`GET /`** — welcome JSON (assignment-style root route)
  - **`GET /health`** — `models_loaded`, `classifier_loaded`, `regressor_loaded` (registry load status)
  - **`POST /predict`** — Pydantic-validated body → LLM feature extraction → sklearn classifier + regressor → win probability + expected monetary outcome
  - **Similar cases** — return top-K similar historical cases with explanations
  - **Counterfactual** — accept feature changes, return outcome deltas
- Load **Production**-stage sklearn models from the MLflow model registry (`models.tracking.load_production_model`, `models:/<name>/Production`)
- Run the feature extraction pipeline on incoming raw case data
- Handle request validation (Pydantic on `/predict` before inference), error responses, and health checks
>>>>>>> origin/phase-1/scraper-and-data

## Endpoints

<<<<<<< HEAD
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health check + model load status |
| POST | `/predict` | Single case prediction |
| POST | `/predict/batch` | Batch predictions (1–50 cases) |
| POST | `/similar` | Find similar historical cases |
| POST | `/counterfactual` | Feature perturbation analysis |

## `/predict` Request/Response

```json
// Request
{
  "case_text": "Plaintiff is suing defendant for $3,500...",
  "claim_amount": 3500,
  "has_attorney": false
}

// Response
{
  "win_probability": 0.73,
  "expected_monetary_outcome": 2800.0,
  "confidence": "high",
  "feature_vector": {...}
}
```

Confidence levels: `"high"` (prob > 0.7 or < 0.3), `"medium"` (0.4–0.6), `"low"` (otherwise).

## Model Loading

On startup, `load_models()` fetches Production-stage models from MLflow registry:
- `litigation-win-classifier`
- `litigation-monetary-regressor`

If models aren't in Production yet, `/health` returns `models_loaded: false` and prediction endpoints return 503.

## Running

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
# or via Docker (see docker/)
```
=======
- Separate services for feature extraction (LLM calls) and model inference (fast)
- Feature extraction has higher latency due to LLM calls — consider async or pre-extraction
- Models are loaded from MLflow — service needs **reachable** `MLFLOW_TRACKING_URI` (and registry access for artifact download)
- Sklearn `predict` / `predict_proba` run in **`asyncio.to_thread`** so async routes do not block the event loop on CPU work
- Must support both batch and real-time inference
- Input validation is critical — this is the system boundary
- Containerized via Docker (see `docker/Dockerfile.inference` and root README)
- Deployed to AWS ECS or Google Cloud Run (see `infra/`)
>>>>>>> origin/phase-1/scraper-and-data
