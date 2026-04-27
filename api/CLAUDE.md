# API Module

FastAPI inference service — the user-facing layer of the pipeline.

## File Map

| File | Purpose |
|---|---|
| `app.py` | FastAPI app, lifespan, and endpoints |
| `dependencies.py` | AppState singleton — loads models, feature extractor, and retrieval index |
| `schemas.py` | Pydantic request/response models |

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Welcome JSON |
| GET | `/health` | Health check and model load status |
| POST | `/predict` | Single case prediction |
| POST | `/predict/batch` | Batch predictions, up to 50 cases |
| POST | `/similar` | Find similar historical cases |
| POST | `/counterfactual` | Feature perturbation analysis |

## Model Loading

On startup, `load_models()` fetches Production-stage models from the MLflow registry:

- `litigation-win-classifier`
- `litigation-monetary-regressor`

If either model is not in Production, `/health` returns `models_loaded: false` and
prediction endpoints return `503`.

## Feature Contract

Training and inference use the shared `v2 feat_*` preprocessing in
`models.dataset`. The API converts LLM-extracted `FeatureVector` values to the
same model columns used by `dataset.csv` training. Missing required inference
features return a validation error instead of being imputed.

## Runtime Notes

- Models are loaded from MLflow, so `MLFLOW_TRACKING_URI` must be reachable.
- Sklearn `predict` and `predict_proba` run in `asyncio.to_thread` so CPU work does not block the event loop.
- Feature extraction calls an LLM and may be slower than model inference.
- Containerized runtime is defined in `docker/Dockerfile.inference`.
