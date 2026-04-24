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

- **LLM for feature extraction only** — not for prediction. The LLM converts raw case text into ~40 **existence-based booleans** (is X stated/attached/named in the text?) plus a few numerics. Subjective 1–5 ratings were retired in the v2 schema in favor of observable facts. Prediction is handled by traditional ML models (classification + regression).
- **Unilateral feature perspective**: features are written as `user_*` / `opposing_party_*`. `user_side` is auto-detected per case from whether a `DEFENDANT_S_CLAIM` document exists; derived `user_is_plaintiff` flag lets one model serve both sides.
- **Leakage firewall**: outcome documents (judgments, orders, dismissals, stipulations) are excluded from feature-extraction text. The exclusion list lives in `features/labels.LABEL_DOC_KEYWORDS` — `FeatureExtractor` input and `LabelExtractor` input must remain disjoint by doc type.
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
- **GPU / Colab:** `notebooks/colab_gpu_extraction.ipynb` — filtered downloads, PyMuPDF + Qwen2-VL, label extraction (Step 5), and v2 **feature extraction** (Step 6) that builds `ProcessedCase` from Drive-mounted txts and writes `features_cache/` back to Drive. Supports `MY_WORKER`/`TOTAL_WORKERS` stride sharding.
- **Labels:** `features/labels.py` — LLM extraction of outcome labels from judgment-style text (separate from feature extraction). Shared `LABEL_DOC_KEYWORDS` used by both label extraction and the leakage filter on the feature-extraction side.
- **Features (v2):** `features/schema.py`, `features/prompts.py`, `features/extraction.py` — existence-based booleans, unilateral perspective, `user_side` threaded through the prompt. `feature_version = "v2"`; older cache JSONs are not compatible.
- **Training / assignment API:** `scripts/train_binary_classifier.py` (synthetic demo → MLflow register), `scripts/promote_models_to_production.py`. FastAPI **`GET /`**, **`GET /health`**, **`POST /predict`** in `api/app.py`; models loaded from **MLflow Production** registry. **README** has full run + Docker instructions; **remote MLflow** needs `--serve-artifacts` and (MLflow 3 UI) **`--cors-allowed-origins`** — see `mlflow/CLAUDE.md`.
