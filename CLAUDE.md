# Litigation Outcome Pipeline

End-to-end ML system for predicting small claims court case outcomes (plaintiff win probability + expected monetary outcome).

## Architecture Overview

```
scraper/ → data/ → features/ → models/ → api/
                                  ↕
                               mlflow/
retrieval/ (embedding similarity + explanations)
counterfactual/ (feature perturbation analysis)
```

## Key Design Decisions

- **LLM for feature extraction only** — not for prediction. The LLM converts raw case text into structured signals (evidence strength, contract presence, argument clarity). Prediction is handled by traditional ML models (classification + regression).
- **MLflow** manages all experiment tracking, metric logging, and model versioning.
- **Retrieval** uses embeddings to find similar historical cases and ground explanations in real examples.
- **Counterfactual analysis** simulates feature changes to show predicted outcome shifts.
- **Dockerized** with separate services for feature extraction and model inference.
- **CI/CD** via GitHub Actions — testing, data validation, Docker builds, deployment.
- **Cloud deployment** targeting AWS (S3 + ECS) or GCP (GCS + Cloud Run).

## Tech Stack

- Python (primary language)
- MLflow (experiment tracking + model registry)
- Docker + Docker Compose (containerization)
- GitHub Actions (CI/CD)
- AWS S3 / GCS (data storage)
- AWS ECS / Cloud Run (serving)

## Conventions

- Each subdirectory has its own CLAUDE.md with component-specific context.
- Tests mirror the source structure under `tests/`.
- All model artifacts are tracked through MLflow — never committed to git.
- Raw data lives in `data/raw/`, processed in `data/processed/`, schemas in `data/schemas/`.

## Phase-1 additions (scraper / data / serving)

- **Court data:** API-based scraper (`scraper/court_api.py`), case **enumeration** (`scrape enumerate`), **`scrape download-cases`**. Session ID is **manual** (Cloudflare). **Document type whitelist** in `scraper/config.py` reduces PDF volume.
- **GPU / Colab:** `notebooks/colab_gpu_extraction.ipynb` — filtered downloads, PyMuPDF + Qwen2-VL, optional label step.
- **Labels:** `features/labels.py` — LLM extraction of outcome labels from judgment-style text (separate from feature extraction).
- **Training:** `scripts/train_classifier_real.py` trains both models from `dataset.csv` using the v2 `feat_*` feature set in `models/dataset.py` — this is the production training path. `scripts/train_binary_classifier.py` is a synthetic-data smoke test only. `scripts/promote_models_to_production.py` promotes registered versions to Production so the API can load them.
- **API:** FastAPI **`GET /`**, **`GET /health`**, **`POST /predict`**, **`POST /similar`**, **`POST /counterfactual`** in `api/app.py`; models loaded from **MLflow Production** registry at `http://35.208.251.175:5000`.
