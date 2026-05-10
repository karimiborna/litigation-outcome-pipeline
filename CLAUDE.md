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
- **GPU / Colab:** `notebooks/colab_gpu_extraction.ipynb` — filtered downloads, PyMuPDF + Qwen2-VL, optional label step.
- **Labels:** `features/labels.py` — LLM extraction of outcome labels from judgment-style text (separate from feature extraction).
- **Training:** `scripts/train_models.py` trains both models from `dataset.csv` using the v2 `feat_*` feature set in `models/dataset.py` — this is the production training path. `scripts/train_binary_classifier.py` is a synthetic-data smoke test only. `scripts/promote_models_to_production.py` promotes registered versions to Production so the API can load them.
- **API:** FastAPI **`GET /`**, **`GET /health`**, **`POST /predict`**, **`POST /similar`**, **`POST /counterfactual`** in `api/app.py`; models loaded from **MLflow Production** registry at `http://35.208.251.175:5000`.

## Roadmap (priority order)

The pipeline runs end-to-end on `dataset.csv`. The remaining work is to make the system actually useful. Tackle in this order:

1. **Perturbation analysis** — `counterfactual/analyzer.py` runs but the v2 path is half-baked: `FEATURE_CONSTRAINTS` is v1-only (so `_clamp` is a no-op on v2), `_auto_perturbations_v2` flips every binary including non-actionable ones (`feat_user_is_plaintiff`, damages-breakdown, jurisdictional), and the curated perturbable set described in `counterfactual/CLAUDE.md` is not implemented in code. Restrict to actionable features and add real v2 constraints.
2. **RAG + LLM explanation layer** — `HybridCaseIndex` (FAISS dense + BM25 sparse + RRF fusion in `retrieval/index.py`) returns ranked cases but nothing consumes them. Add an LLM step that takes top-K similar cases (with outcomes joined from `dataset.csv`) plus the top counterfactual deltas and produces a single grounded narrative explaining how retrieved cases relate to the user's case and what the perturbation deltas mean. The `reranker` slot on `HybridCaseIndex` is wired but unused — decide whether to use it for cross-encoder reranking or drop it.
3. **Model optimization** — hyperparameter tuning, probability calibration, proper CV beyond `scripts/train_models.py` defaults.
4. **Feature selection** — empirically pare `MODEL_FEATURE_COLUMNS` (currently 51 columns, including 8 one-hot category dummies) down to features that actually move metrics.
5. **Missing feature verification** — audit gaps between the documented schema and `RAW_MODEL_FEATURE_COLUMNS`. Known gap: contract-detail booleans (`contract_is_written`, `contract_is_signed_by_both_parties`, `contract_specifies_deadline_or_term`, `contract_specifies_payment_amount`) are in `FeatureVector` but absent from training columns. Decide whether to include or remove from the schema.
