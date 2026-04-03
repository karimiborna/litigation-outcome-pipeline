# Docker Module

Two Docker images — one for LLM-heavy feature extraction, one for lightweight inference.

## Images

**`Dockerfile.features`** — Feature extraction service
- Heavier image (LLM dependencies)
- Runs the feature extraction pipeline
- Requires NVIDIA_API_KEY

**`Dockerfile.inference`** — Inference service
- Lighter image (FastAPI + scikit-learn)
- Serves the API endpoints
- Loads models from MLflow at startup

## Local Development

```bash
docker compose up
# Feature service: http://localhost:8001
# Inference API:  http://localhost:8000
# MLflow UI:      http://localhost:5000
```

Env vars are loaded from `.env` file (never baked into images).

## docker-compose.yml

Runs three services:
1. `mlflow` — tracking server
2. `features` — LLM feature extraction
3. `api` — FastAPI inference
