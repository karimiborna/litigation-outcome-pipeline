# Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION                               │
│                                                                     │
│  scraper/cli.py (Click CLI)                                         │
│    ├── scrape        → court_api.py → SF Superior Court DataSnap API│
│    ├── enumerate     → enumerator.py → brute-force case discovery   │
│    ├── download-cases→ court_api + extractor.py (PDF download)      │
│    └── extract       → extractor.py (pymupdf → NVIDIA vision OCR)  │
│                                                                     │
│  Session: manual Cloudflare CAPTCHA → env var                       │
│  Rate limiter: 2.5s delay, 200 req/day cap                         │
│  State: manifest.json + valid_cases.json (resume support)           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                   │
│                                                                     │
│  data/raw/<case>/       → metadata.json + PDFs (immutable)          │
│  data/processed/<case>/ → extracted .txt files                      │
│  data/schemas/case.py   → CaseMetadata, ProcessedCase (Pydantic)    │
│  data/cleaning.py       → OCR artifact removal, whitespace collapse │
│  data/validation.py     → schema validation at stage boundaries     │
│  data/storage.py        → filesystem read/write helpers             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  FEATURE EXTRACTION (LLM)                           │
│                                                                     │
│  features/extraction.py → FeatureExtractor                          │
│    - Sends case text to LLM (OpenAI-compatible API)                 │
│    - Parses JSON response → LLMFeatures (~40 existence-based        │
│      booleans: has_*, argument_*, sent_*, damages_*, contract_*,    │
│      plus claim_category, monetary_amount_claimed, counts)          │
│    - Unilateral: user_has_attorney vs opposing_party_has_attorney;  │
│      user_side threaded from ProcessedCase into the prompt          │
│    - Merges with derived counts (text_length, document_count,       │
│      user_is_plaintiff) → FeatureVector                             │
│    - SHA256-keyed disk cache (idempotent); feature_version = "v2"   │
│    - Leakage firewall: label docs excluded from input text          │
│                                                                     │
│  features/labels.py → separate LLM pipeline for outcome labels      │
│  features/prompts.py → prompt templates                             │
│  features/schema.py  → FeatureVector.to_model_input() → dict       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
┌──────────────────┐ ┌──────────┐ ┌─────────────────────┐
│  ML MODELS       │ │ RETRIEVAL│ │ COUNTERFACTUAL      │
│                  │ │          │ │                     │
│ GBClassifier     │ │ sentence-│ │ Perturb 1 feature   │
│  → win/loss prob │ │ transformers│  at a time        │
│ GBRegressor      │ │ (MiniLM) │ │ Re-run classifier   │
│  → $ amount      │ │ + FAISS  │ │  + regressor        │
│                  │ │ index    │ │ Sort by |Δ win prob| │
│ MLflow tracked   │ │          │ │                     │
│ MLflow registry  │ │ cosine   │ │ "If you had a       │
│  (Production     │ │ similarity│ │  contract, +27%"   │
│   stage)         │ │ top-K=5  │ │                     │
└────────┬─────────┘ └────┬─────┘ └──────────┬──────────┘
         │                │                   │
         └────────────────┼───────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FastAPI  (api/app.py)                          │
│                                                                     │
│  GET  /           → welcome                                         │
│  GET  /health     → model load status                               │
│  POST /predict    → case text → LLM features → classifier/regressor │
│  POST /predict/batch → up to 50 cases                               │
│  POST /similar    → FAISS search → top similar cases                │
│  POST /counterfactual → "what-if" feature perturbation              │
│                                                                     │
│  Startup: loads models from MLflow Production + FAISS index          │
│  sklearn predict runs in asyncio.to_thread (non-blocking)           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     INFRA / DEPLOYMENT                              │
│                                                                     │
│  docker-compose.yml → 3 services:                                   │
│    1. mlflow        (port 5000, sqlite backend)                     │
│    2. feature-service (port 8001, Dockerfile.features)              │
│    3. inference-service (port 8000, Dockerfile.inference)            │
│                                                                     │
│  infra/ → Terraform (AWS ECS + S3) — defined but not deployed       │
│  .github/workflows/ → CI, data validation, Docker build, deploy     │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Status

| Component | Status |
|---|---|
| **Scraper** (CLI, court API, enumeration, PDF download, OCR) | Built and functional |
| **Data layer** (schemas, storage, cleaning, validation) | Built |
| **Feature extraction** (LLM calls, caching, schema) | Built, code complete |
| **ML models** (trainer, MLflow tracking/registry) | Code built, trained on **synthetic data** only |
| **Retrieval** (embeddings, FAISS index) | Code built, needs real case index |
| **Counterfactual** (feature perturbation) | Code built, works once models exist |
| **API** (all 5 endpoints) | Code built, works end-to-end if models are in MLflow Production |
| **Docker** (compose + 2 Dockerfiles) | Defined |
| **Infra** (Terraform for AWS ECS/S3) | Defined, not deployed |
| **CI/CD** (GitHub Actions) | Workflows exist for CI, data validation, Docker build, deploy |

## Main Gap

The pipeline is structurally complete end-to-end, but models are trained on synthetic data. The next step is getting enough real scraped + extracted + labeled cases to retrain on real data and build a real FAISS index.
