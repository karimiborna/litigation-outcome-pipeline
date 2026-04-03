# API Module

FastAPI inference service — the user-facing layer of the pipeline.

## File Map

| File | Purpose |
|---|---|
| `app.py` | FastAPI app, lifespan, all endpoints |
| `dependencies.py` | AppState singleton — loads models/index on startup |
| `schemas.py` | Pydantic request/response models |

## Endpoints

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
