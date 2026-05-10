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

### Recently shipped

- **Perturbation analysis (v2)** — `counterfactual/analyzer.py` now operates on a curated 28-feature set (only what a litigant can actually change), real v2 `FEATURE_CONSTRAINTS`, batched predicts, helpful/harmful direction tagging, witness-count stepping with early-stop, and a `select_top_recommendations` helper.
- **LLM advice grounded in perturbations** — the existing similarity-advice + judge LLM calls now receive a `format_for_llm` summary of the top-5 perturbations and are instructed to reference them, reconcile them with retrieved cases, and treat load-bearing flips as warnings. The `LexRatioAnalysisResponse` carries both the prose `advice` and a structured `top_recommendations: list[CounterfactualItem]` for the frontend.

### Open, in priority order

1. **Surface `top_recommendations` in the LexRatio frontend** — the data ships in `/api/analyze-lexratio` but `lexratio.html` doesn't render the top-5 yet.
2. **Model optimization** — hyperparameter tuning, probability calibration, proper CV beyond `scripts/train_models.py` defaults.
3. **Feature selection** — empirically pare `MODEL_FEATURE_COLUMNS` (currently 51 columns, including 8 one-hot category dummies) down to features that actually move metrics.
4. **Missing feature verification** — audit gaps between the documented schema and `RAW_MODEL_FEATURE_COLUMNS`. Known gap: contract-detail booleans (`contract_is_written`, `contract_is_signed_by_both_parties`, `contract_specifies_deadline_or_term`, `contract_specifies_payment_amount`) are in `FeatureVector` but absent from training columns. Decide whether to include or remove from the schema.
5. **Decide on the `HybridCaseIndex.reranker` slot** — wired but unused. Either plug in a cross-encoder reranker or drop the slot.
