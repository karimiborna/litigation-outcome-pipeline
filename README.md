# Litigation Outcome Pipeline

An end-to-end machine learning system for predicting the outcomes of small claims court cases. Given raw case data scraped from the SF Superior Court, the system produces two predictions: **the probability of a plaintiff win** and **the expected monetary outcome**.

The core design principle: the LLM is used strictly for **feature extraction**, not prediction. Raw court documents (scanned PDFs) are run through OCR, then an LLM converts the unstructured text into structured signals — evidence strength, contract presence, argument clarity, and more. These signals are merged with case metadata (claim amount, party count, legal representation) into a unified feature matrix, which feeds into traditional scikit-learn classification and regression models. All model experiments are tracked and versioned through MLflow.

Beyond prediction, the system provides two interpretability modules:

- **Similar case retrieval** — embeds case text with sentence-transformers, builds a FAISS vector index, and returns the most similar historical cases to ground explanations in real examples
- **Counterfactual analysis** — perturbs individual features and re-runs predictions to show how specific changes (e.g., "what if the plaintiff had stronger evidence?") would shift the outcome

The full system is containerized with Docker, deployed to AWS ECS via Terraform, and automated with GitHub Actions CI/CD.

## Architecture

```
scraper → data → features → models → api
                                ↕
                             mlflow

          retrieval ←──── api ────→ counterfactual

          docker + infra + .github/workflows wrap the whole thing
```

## Current State

### What's Built

- **Scraper** — fully working. Uses a reverse-engineered DataSnap REST API (JSON endpoints) to fetch cases and documents from the SF Superior Court. PDF text extraction uses PyMuPDF for text-based PDFs (free, instant) with NVIDIA vision API fallback for scanned PDFs.
- **Case enumerator** — brute-forces case number ranges to discover historical cases that are no longer on the court calendar.
- **Data layer** — Pydantic schemas, validation, text cleaning, and filesystem storage.
- **Features module** — LLM prompt templates, feature schemas, and extraction logic (ready to use once data is collected).
- **Models module** — scikit-learn trainers with MLflow tracking (ready to use once features are extracted).
- **Retrieval module** — sentence-transformers + FAISS index for similar case search.
- **Counterfactual module** — feature perturbation analysis.
- **API** — FastAPI endpoints for prediction, retrieval, and counterfactual analysis.
- **Docker, Infra, CI/CD** — Dockerfiles, Terraform (AWS), and GitHub Actions workflows.
- **Tests** — 110 unit tests, all passing.

### Data Collection Progress

The range `CSM25870000` → `CSM25879999` has been probed for the first 100 case numbers — all 100 were valid (have documents). Enumeration needs to continue through the rest of that range and through `CSM26870000` → `CSM26879999`. This can be split across team members since the enumerator supports resume (re-running skips already-probed numbers).

```bash
# Continue from where we left off (skips the first 100 automatically)
scrape enumerate --start CSM25870000 --end CSM25879999

# Another team member can do the second range in parallel
scrape enumerate --start CSM26870000 --end CSM26879999
```

After enumeration, run `scrape download-cases` to download PDFs for all valid cases.

### GPU Text Extraction (Google Colab)

The notebook at `notebooks/colab_gpu_extraction.ipynb` handles the full pipeline on Google's servers — no files need to leave your laptop, and no NVIDIA API quota is consumed.

1. Open the notebook in Colab and set **Runtime → Change runtime type → T4 GPU**
2. It clones the repo (gets `valid_cases.json` with all known case numbers)
3. Downloads PDFs directly from the court API into Colab (prompts you for a session ID)
4. Extracts text: PyMuPDF first (free, instant), then **Qwen2-VL-7B** on GPU for scanned pages
5. Saves to Google Drive (persistent) and/or lets you download a zip

With `USE_DRIVE = True`, PDFs and text files are saved to your Google Drive so they persist if the Colab runtime disconnects. Set it to `False` to skip Drive and just download the zip at the end.

### What Needs to Happen Next

1. **Finish enumeration + download** — Complete the remaining case number ranges and run `scrape download-cases` to grab PDFs. Also run `scrape scrape --date <date>` for recent calendar dates.
2. **Extract features** — Once enough cases are collected, run the LLM feature extraction on the extracted text (requires `LLM_API_KEY` in `.env`).
3. **Label data** — Determine case outcomes (plaintiff win/loss, monetary result) from the extracted text. This may require manual review or additional parsing.
4. **Train models** — Use the features module output to train the classifier and regressor via MLflow.
5. **Build retrieval index** — Embed all case texts and build the FAISS index.
6. **Deploy** — Containerize and deploy to AWS ECS.

### All CLI Commands

```bash
# Scrape cases from the court calendar (recent dates only)
scrape scrape --date 2026-04-01
scrape scrape --start-date 2026-03-01 --end-date 2026-04-01
scrape scrape --days-back 30
scrape scrape --date 2026-04-01 --no-extract    # skip text extraction

# Enumerate historical cases by brute-forcing case number ranges
scrape enumerate --start CSM25870000 --end CSM25879999
scrape enumerate --start CSM26870000 --end CSM26879999 --delay 0.5

# Download PDFs for all valid cases found by enumerate
scrape download-cases
scrape download-cases --no-extract

# Extract text from already-downloaded PDFs for a specific case
scrape extract --case-number CSM26871146

# Check scraping progress
scrape status
```

All commands prompt you for a session ID on startup (see Setup below).

## Setup

### Prerequisites

- **Python 3.12** (3.9+ works for everything except Docker builds)
- **An NVIDIA API key** (optional) — only needed for scanned PDFs that don't have selectable text. Most court PDFs have selectable text and skip the API entirely.
- **An OpenAI API key** (or compatible provider) — needed later for LLM feature extraction, not for scraping.

### Installation

```bash
cd litigation-outcome-pipeline

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the project and all dependencies
pip install -e ".[dev]"
```

This installs the project in editable mode, registers the `scrape` CLI command, and pulls in dev tools (pytest, ruff, mypy).

### Environment Configuration

```bash
cp .env.example .env
```

Edit `.env` and fill in what you need:

| Variable | Required For | Where to get it |
|---|---|---|
| `SFTC_SESSION_ID` | Scraping (or paste in terminal when prompted) | Visit `https://webapps.sftc.org/cc/CaseCalendar.dll` in your browser and copy the hex string after `SessionID=` in the URL |
| `NVIDIA_API_KEY` | Scanned PDF extraction | [NVIDIA Build](https://build.nvidia.com) — free tier, ~500 req/day |
| `LLM_API_KEY` | Feature extraction (later) | [OpenAI API Keys](https://platform.openai.com/api-keys) |

The session ID expires after ~10 minutes of **inactivity**, but stays alive as long as the scraper is making requests. If you leave `SFTC_SESSION_ID` blank, the scraper will prompt you to paste one in the terminal when it starts.

### Getting a Session ID

1. Open `https://webapps.sftc.org/cc/CaseCalendar.dll` in your browser
2. The URL will redirect to something like `...?=&SessionID=4D16483901435613A43C5AD805EEBBF08F6437DB`
3. Copy that hex string
4. Either paste it in the terminal when prompted, or put it in `.env` as `SFTC_SESSION_ID=<value>`

Sessions are Cloudflare-issued and require a browser — there is no way to acquire them programmatically.

## Usage

### Scraping Court Data

The scraper calls the SF Superior Court's DataSnap REST API to fetch case listings and documents, downloads PDFs, and extracts text using PyMuPDF (with NVIDIA vision API fallback for scanned documents).

**Calendar-based scraping** (recent dates only — the calendar purges older dates):

```bash
scrape scrape --date 2026-04-01
scrape scrape --start-date 2026-03-01 --end-date 2026-04-01
scrape scrape --date 2026-04-01 --no-extract   # PDFs only, no text extraction
```

**Case number enumeration** (for historical data not on the calendar):

```bash
# Phase 1: probe case numbers to find which ones exist
scrape enumerate --start CSM25870000 --end CSM25879999

# Phase 2: download PDFs for all valid cases found
scrape download-cases
```

The enumerator probes each case number at 1 request/second and saves valid ones to `scraper/state/valid_cases.json`. It supports resume — re-running skips already-probed numbers.

**The scraper is fully resumable.** If interrupted, running the same command again picks up where it left off via the manifest at `scraper/state/manifest.json`.

**Respectful scraping is enforced:** 2.5-second delay between requests, 200 requests/day cap (configurable), academic research User-Agent header, and session reuse.

### Where Data Goes

```
data/
├── raw/
│   └── pdfs/                     ← downloaded court PDFs (immutable)
│       ├── CSM26871146_CLAIM_OF_PLAINTIFF.pdf
│       ├── CSM26871204_PROOF_OF_SERVICE_ON_CLAIM.pdf
│       └── ...
└── processed/
    └── extracted/                ← extracted text (one .txt per PDF)
        ├── CSM26871146_CLAIM_OF_PLAINTIFF.txt
        ├── CSM26871204_PROOF_OF_SERVICE_ON_CLAIM.txt
        └── ...

scraper/state/
├── manifest.json                 ← tracks scraped dates and cases (resume support)
└── valid_cases.json              ← case numbers found by enumerate
```

Raw data is **immutable** — all transformations produce new files in `processed/`. This ensures the original scraped data is always available for re-processing.

### Feature Extraction

Once text has been extracted from PDFs, the features module sends it to an LLM to produce structured signals:

| Feature | Type | Description |
|---|---|---|
| `evidence_strength` | 1–5 int | Quality and quantity of supporting evidence |
| `contract_present` | bool | Whether a written contract is referenced |
| `argument_clarity_plaintiff` | 1–5 int | Coherence of plaintiff's arguments |
| `argument_clarity_defendant` | 1–5 int | Coherence of defendant's arguments |
| `claim_category` | string | E.g., unpaid_debt, property_damage, breach_of_contract |
| `monetary_amount_claimed` | float | Dollar amount claimed |
| `prior_attempts_to_resolve` | bool | Evidence of prior negotiation or demand letters |
| `witness_count` | int | Number of witnesses mentioned |
| `documentary_evidence` | bool | Whether documents/photos/receipts are referenced |
| `timeline_clarity` | 1–5 int | How clear the timeline of events is |
| `legal_representation_plaintiff` | bool | Whether plaintiff has an attorney |
| `legal_representation_defendant` | bool | Whether defendant has an attorney |
| `counterclaim_present` | bool | Whether the defendant filed a counterclaim |
| `default_judgment_likely` | bool | Whether defendant appears absent/non-responsive |

These are merged with metadata-derived features (plaintiff count, defendant count, text length, document count) into a `FeatureVector` that becomes the model input.

The LLM prompt enforces JSON-only output with no explanation text. Feature extraction is **idempotent** and **cached** — the same input always produces the same features, and results are stored in `data/features_cache/` to avoid redundant LLM calls.

### Model Training

Two models are trained on the feature matrix:

1. **Classifier** — Gradient Boosting Classifier predicting plaintiff win probability
2. **Regressor** — Gradient Boosting Regressor predicting expected monetary outcome

Both use scikit-learn's `GradientBoostingClassifier`/`GradientBoostingRegressor` with the following defaults: 200 estimators, max depth 5, learning rate 0.1, fixed random state for reproducibility.

Every training run is logged to MLflow:
- All hyperparameters
- Evaluation metrics (accuracy, precision, recall, F1, ROC-AUC for the classifier; MAE, RMSE, R² for the regressor)
- Top-10 feature importances
- The model artifact itself, registered in the MLflow model registry

Models are **never committed to git** — they live exclusively in MLflow's artifact store.

### MLflow

MLflow is the tracking backbone for all experimentation. Start the local server:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root mlruns/artifacts \
  --host 0.0.0.0 --port 5000
```

Or use Docker Compose (see below), which starts MLflow automatically.

The MLflow UI at `http://localhost:5000` shows all experiments, metrics, and registered models. Models progress through registry stages: **None → Staging → Production**. The API loads whichever model version is in the Production stage.

### Similar Case Retrieval

The retrieval module uses `sentence-transformers` (default model: `all-MiniLM-L6-v2`) to embed case text into dense vectors, then builds a FAISS `IndexFlatIP` (inner product / cosine similarity) index over all historical cases.

Given a new case, it returns the top-K most similar historical cases with similarity scores. Results are filtered by a configurable similarity threshold (default: 0.3). The index persists to disk at `data/retrieval_index/` and needs to be rebuilt when new historical data is added.

### Counterfactual Analysis

Given a case's feature vector, the counterfactual module:

1. Runs the original features through both models to get baseline predictions
2. For each perturbable feature, generates a meaningful change (e.g., evidence strength 3 → 4, contract present false → true)
3. Re-runs the modified features through both models
4. Reports the delta: "If evidence strength changed from 3 to 5, win probability would increase by 12.3%"

Feature perturbations respect constraints — evidence strength stays between 1–5, monetary amounts can't go negative, booleans flip between 0 and 1. Results are sorted by impact magnitude so the most influential changes appear first.

Custom perturbations can also be passed explicitly via the API.

### API

The API is a FastAPI application exposing four endpoint groups:

| Endpoint | Method | Description |
|---|---|---|
| `GET /health` | GET | Health check — reports status, version, and whether models are loaded |
| `POST /predict` | POST | Single case prediction — accepts case text, returns win probability + expected monetary outcome |
| `POST /predict/batch` | POST | Batch prediction — up to 50 cases at once |
| `POST /similar` | POST | Similar case retrieval — accepts case text, returns top-K similar historical cases |
| `POST /counterfactual` | POST | Counterfactual analysis — accepts case text and optional perturbations, returns outcome deltas |

On startup, the API loads production models from MLflow, initializes the feature extractor (LLM client), and loads the FAISS retrieval index. If any component fails to load, the corresponding endpoint returns a 503.

Run locally:

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Docker

Two separate Docker images keep concerns isolated and allow independent scaling:

| Image | Dockerfile | Purpose | Port |
|---|---|---|---|
| Feature extraction service | `docker/Dockerfile.features` | Handles LLM calls and NVIDIA OCR (heavier, GPU-friendly) | 8001 |
| Inference / API service | `docker/Dockerfile.inference` | Model serving, retrieval, counterfactual analysis (lightweight, fast) | 8000 |

Both use multi-stage builds to keep images small. Secrets are passed via environment variables, never baked into images.

Run everything locally with Docker Compose:

```bash
cd docker
docker compose up --build
```

This starts three services:
- **mlflow** on port 5000 — experiment tracking UI and model registry
- **feature-service** on port 8001 — LLM-based feature extraction
- **inference-service** on port 8000 — the main API

### Infrastructure (AWS)

The `infra/` directory contains Terraform configuration for deploying to AWS:

- **S3 buckets** — one for case data (`raw/` and `processed/`), one for MLflow artifacts. Both have versioning, encryption, and public access blocked.
- **ECS Fargate** — serverless container hosting for both the inference and feature extraction services. Separate task definitions with independent CPU/memory allocations.
- **IAM** — execution roles for ECS and task roles with scoped S3 access.
- **CloudWatch** — log groups with 30-day retention for both services.

Separate variable files for environments:

```bash
# Deploy to dev
terraform apply -var-file=infra/envs/dev.tfvars

# Deploy to production
terraform apply -var-file=infra/envs/prod.tfvars
```

Dev uses minimal resources (256 CPU / 512MB memory, 1 task). Production scales up (1024 CPU / 2048MB, 2 tasks).

### CI/CD

Four GitHub Actions workflows automate the pipeline:

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | Push/PR to main | Runs ruff lint + format checks, pytest on Python 3.10 and 3.12, mypy type checking |
| `data-validation.yml` | Changes to `data/schemas/` | Validates that all Pydantic schemas are importable and schema tests pass |
| `docker-build.yml` | Push to main | Builds both Docker images, tags with git SHA, pushes to GitHub Container Registry |
| `deploy.yml` | Manual trigger | Deploys to AWS ECS (dev or production), waits for service stability |

Tests must pass before merge. Docker images are tagged with both `latest` and the git SHA for traceability.

## Project Structure

```
litigation-outcome-pipeline/
├── scraper/                    # Court site scraper + PDF text extraction
│   ├── cli.py                  # Click CLI (scrape, enumerate, download-cases, extract, status)
│   ├── config.py               # ScraperConfig from env vars + API constants
│   ├── session.py              # Session ID management (manual paste via terminal)
│   ├── court_api.py            # DataSnap REST API client (get_cases, get_documents, get_roa)
│   ├── court_scraper.py        # Main orchestrator for calendar-based scraping
│   ├── enumerator.py           # Case number brute-force enumeration
│   ├── extractor.py            # PDF text extraction (PyMuPDF + NVIDIA vision fallback)
│   ├── manifest.py             # Resume support (tracks what's been scraped)
│   └── rate_limiter.py         # Request delays + daily caps
├── data/
│   ├── schemas/case.py         # Pydantic models: CaseMetadata, ExtractedText, ProcessedCase
│   ├── validation.py           # Schema validation utilities
│   ├── cleaning.py             # OCR text cleaning (unicode, artifacts, whitespace)
│   └── storage.py              # Filesystem save/load for metadata, PDFs, text
├── features/
│   ├── config.py               # LLM provider settings
│   ├── prompts.py              # Prompt templates for structured feature extraction
│   ├── schema.py               # LLMFeatures + FeatureVector (model input)
│   └── extraction.py           # LLM client, response parsing, caching
├── models/
│   ├── config.py               # MLflow tracking/registry settings
│   ├── tracking.py             # MLflow helpers (experiments, logging, registry)
│   └── trainer.py              # ClassifierTrainer + RegressorTrainer
├── retrieval/
│   ├── config.py               # Embedding model + index settings
│   ├── embeddings.py           # Sentence-transformer wrapper
│   └── index.py                # FAISS index build/search/persist
├── counterfactual/
│   └── analyzer.py             # Feature perturbation + outcome delta analysis
├── api/
│   ├── app.py                  # FastAPI application with all endpoints
│   ├── schemas.py              # Request/response Pydantic models
│   └── dependencies.py         # Model loading + shared state
├── mlflow/
│   └── server_config.yaml      # MLflow server configuration
├── docker/
│   ├── Dockerfile.features     # Feature extraction service image
│   ├── Dockerfile.inference    # Inference/API service image
│   └── docker-compose.yml      # Local multi-service development
├── infra/
│   ├── main.tf                 # Terraform provider + backend
│   ├── variables.tf            # All configurable variables
│   ├── s3.tf                   # Data + MLflow artifact buckets
│   ├── ecs.tf                  # ECS cluster, task definitions, services
│   ├── outputs.tf              # Exported resource identifiers
│   └── envs/                   # dev.tfvars, prod.tfvars
├── .github/workflows/
│   ├── ci.yml                  # Test + lint on every push/PR
│   ├── data-validation.yml     # Schema validation on data/ changes
│   ├── docker-build.yml        # Build + push Docker images on merge
│   └── deploy.yml              # Manual deployment to AWS ECS
├── notebooks/
│   └── colab_gpu_extraction.ipynb  # Google Colab notebook for GPU-based text extraction
├── tests/unit/                 # 110 unit tests across 13 test files
├── pyproject.toml              # Project config, dependencies, tool settings
├── .env.example                # Template for environment variables
└── .gitignore                  # Python, data, MLflow, Docker, IDE ignores
```

## Running Tests

```bash
# All unit tests
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=scraper --cov=data --cov=features --cov=models --cov=api --cov=retrieval --cov=counterfactual

# Lint
ruff check .

# Format check
ruff format --check .

# Type check
mypy scraper/ data/ features/ models/ api/ retrieval/ counterfactual/ --ignore-missing-imports
```

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| ML Models | scikit-learn (GradientBoosting) |
| Experiment Tracking | MLflow |
| Feature Extraction | OpenAI API (gpt-4o-mini) |
| PDF Text Extraction | PyMuPDF + NVIDIA Vision API (Llama 3.2 90B) / Qwen2-VL-7B on Colab GPU |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Search | FAISS |
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Court Data Access | Reverse-engineered DataSnap REST API (requests) |
| Containerization | Docker + Docker Compose |
| Infrastructure | Terraform + AWS (S3, ECS Fargate) |
| CI/CD | GitHub Actions |
| Linting | Ruff |
| Type Checking | mypy |
| Testing | pytest |
