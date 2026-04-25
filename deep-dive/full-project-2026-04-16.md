# Deep Dive: Full Project — Litigation Outcome Pipeline

**Date**: 2026-04-16
**Scope**: Every major component

---

## What This System Is

A machine learning system that predicts whether a plaintiff will win a small claims court case in San Francisco — and by how much. The user inputs their case details and gets:

1. Win probability + confidence level
2. Expected monetary outcome
3. Top similar historical cases (with real outcomes)
4. "What-if" analysis — which changes would most improve their odds

The target user is a **self-represented litigant** (no lawyer). The output has to be explainable and actionable, not just a number.

---

## Architecture Overview

```
SF Court Website
      │ HTTP scraping
      ▼
scraper/              ← Downloads PDFs, extracts text
      │ ProcessedCase
      ▼
features/             ← LLM converts text → 14 structured features
      │ FeatureVector
      ├──→ models/    ← sklearn trains classifier + regressor
      │         │ MLflow artifacts
      │         ▼
      │      api/     ← FastAPI serves predictions
      │         ├── retrieval/      ← FAISS finds similar cases
      │         └── counterfactual/ ← "what if" analysis
      │
      └──→ features/labels.py ← separate LLM pipeline for ground truth labels
```

**The critical design decision**: LLM is used *only* for reading and structuring text. Prediction is done by a traditional sklearn model. This separation matters because LLMs can't give calibrated probabilities (they don't know what they don't know), but sklearn gradient boosting can.

---

## Component-by-Component Walkthrough

---

### 1. Scraper — `scraper/`

**The problem**: SF Superior Court posts case documents at `webapps.sftc.org`. The site uses Cloudflare for bot protection and a DataSnap session ID that rotates. You can't scrape it without a live browser session.

**The solution**: Manual session acquisition + automated everything else.

#### `court_api.py` — HTTP Client

```python
def get_cases(session_id, date_str, config):
    url = f"{BASE_URL}{CASE_PATH}?SessionID={session_id}&date={date_str}"
    response = requests.get(url, headers={"User-Agent": USER_AGENT})
    ...
```

**What**: Wraps the DataSnap REST API with three functions: `get_cases()` (list of cases for a date), `get_documents()` (documents for a case), `download_pdf()` (save one PDF to disk).

**Why a dedicated HTTP client module**: Separating HTTP logic from business logic means you can swap the API client (e.g., mock it in tests) without touching the orchestration code. This is the **Dependency Inversion Principle** in practice.

**Why a custom `User-Agent`**: `"MSDS603-Research-Scraper/1.0 (SF Small Claims academic study)"` — identifies the scraper as academic research, not a bot. Responsible scraping practice.

#### `rate_limiter.py` — Token Bucket

```python
DOWNLOAD_DELAY = 2.5  # seconds between requests
```

**What**: Enforces a minimum delay between requests and a daily cap (200 requests/day).

**Why 2.5 seconds**: Aggressive enough to be practical, slow enough not to appear as a DoS attack. The court's server is a government system — being a good citizen is both ethical and necessary.

**The token bucket pattern**: A token bucket allows a fixed rate of requests. Unlike a simple sleep, it handles bursts gracefully. If you're idle for 10 seconds, you don't "earn" extra tokens — you just get one token per interval. This prevents accidental rate-limit violations after a pause.

#### `manifest.py` — Resume Support

**What**: A JSON file that tracks which dates have been scraped, which cases downloaded, and how many PDFs extracted.

**Why this matters**: Scraping thousands of cases takes days. If the process crashes at 3am, you don't want to restart from zero. The manifest lets you pick up exactly where you left off.

**The pattern (idempotent checkpointing)**: Write progress to a durable log before marking work "done." This is the same pattern used in distributed systems (Kafka consumer offsets, Spark checkpoints). Local manifestation of a universal reliability pattern.

#### `enumerator.py` — Case Discovery

**What**: Brute-forces case numbers by probing the API with number ranges (e.g., CSM25870000–CSM25879999).

**Why necessary**: There's no public index of all cases. You have to guess case numbers and see which ones exist. The 4-digit suffix space is small enough to enumerate.

#### `extractor.py` — Two-Pass PDF Extraction

```python
def extract_text(pdf_path):
    text = extract_with_pymupdf(pdf_path)
    if len(text.strip()) < MIN_TEXT_LENGTH:
        text = extract_with_nvidia(pdf_path)  # vision model fallback
    return text
```

**Pass 1 — PyMuPDF**: Reads the embedded text layer in the PDF. Instant, free, perfect for digital documents (PDF forms filled on a computer).

**Pass 2 — NVIDIA Vision API**: Sends page images to `meta/llama-3.2-90b-vision-instruct` via the NVIDIA API. Used only for scanned paper documents where the text layer is empty.

**Why the 100-character threshold**: A legitimate court document has at least 100 characters. Less than that means the PDF is a scan. This threshold avoids paying for vision API calls on documents that PyMuPDF handles fine.

**NVIDIA daily cap (`NVIDIA_DAILY_CAP = 490`)**: Free tier allows ~500 calls/day per API key. The cap prevents accidentally burning the quota. With 3 team members, that's ~1,470 scanned pages/day across all keys.

---

### 2. Data — `data/`

**The problem**: Data moves between many pipeline stages, and each stage needs to trust the data it receives.

**The solution**: Pydantic schemas at every boundary.

#### `schemas/case.py` — Core Data Models

```
CaseMetadata → ProcessedCase → FeatureVector → model input (dict of floats)
```

Each model represents data at a different pipeline stage:

| Model | Stage | What it holds |
|---|---|---|
| `CaseMetadata` | After scraping | Parties, attorneys, proceedings, document list |
| `ExtractedText` | After PDF extraction | Raw text per document, extraction method |
| `ProcessedCase` | Ready for LLM | Combined text, claim amount, party counts |
| `FeatureVector` | After LLM extraction | 14 numeric/categorical features + metadata |
| `CaseLabels` | Supervision | outcome string, dollar amounts, attorney flags |

**Why this progression**: Each transformation is explicit. You can audit exactly what data looked like at each stage. If your model is wrong, you can trace whether the bug is in scraping, extraction, feature engineering, or training.

**Why `data/raw/` is immutable**: Raw PDFs and metadata are never modified after download. Transformations go to `data/processed/`. This mirrors the **lambda architecture** principle: immutable event log + derived views. If your feature extraction prompt improves, you re-derive features without re-scraping.

#### `storage.py` — Filesystem Layer

```python
def save_metadata(metadata: CaseMetadata, base_dir: Path):
    path = base_dir / "raw" / metadata.case_number / "metadata.json"
    path.write_text(metadata.model_dump_json(indent=2))
```

**Why JSON files, not a database**: SQL databases require infrastructure and migrations. For a research project with thousands of cases, JSON files in a structured directory are:
- Portable (copy the folder, you have all the data)
- Inspectable (open any file in a text editor)
- Git-friendly for small datasets
- Easy to migrate to S3 (same path structure works in cloud storage)

---

### 3. Features — `features/`

**The problem**: ML models need numbers. Court documents are unstructured text.

**The solution**: Use an LLM as a "reading comprehension" layer to extract structured signals.

#### `extraction.py` — Feature Extractor

```python
class FeatureExtractor:
    def extract(self, case: ProcessedCase) -> FeatureVector | None:
        cached = self._load_cache(case)
        if cached:
            return cached
        features = await self._call_llm(case)
        self._save_cache(case, features)
        return features
```

**The cache-first pattern**: Always check cache before calling the LLM. Content-addressable keys (SHA256 of case text + feature version) mean cache invalidation is automatic when either input changes.

**Why version the features**: The feature version string (e.g., `"v1.2"`) is included in the cache key. When you change the extraction prompt (which changes what features mean), the version bumps, old cache entries are ignored, and re-extraction happens automatically. This prevents contaminating your training data with features extracted by different prompt versions.

#### `schema.py` — `FeatureVector.to_model_input()`

```python
def to_model_input(self) -> dict[str, float]:
    return {
        "evidence_strength": float(self.evidence_strength or -1),
        "contract_present": float(self.contract_present) if ... else -1.0,
        ...
    }
```

**Why `-1.0` for nulls (not `0.0` or NaN)**: 
- `NaN` would break sklearn — gradient boosting can't handle NaN without special handling
- `0.0` would be ambiguous — is `contract_present = 0.0` "no contract" or "unknown"?
- `-1.0` is outside the natural range of most features (scales are 1–5, booleans are 0–1), so the model can learn to treat it as a distinct "unknown" category

This is the **sentinel value pattern** — use an out-of-range value to signal "missing," and let the model learn from it.

#### `prompts.py` — Extraction Prompt Design

The prompt:
1. Sets the role: "You are a legal analyst AI"
2. Provides the case context (number, title, cause of action, text)
3. Specifies exact JSON output format with field names and types
4. Forbids markdown, explanation, or extra text

**Why `temperature=0.0`**: Feature extraction needs to be **deterministic and consistent**. The same document should always produce the same features. Temperature=0 removes randomness from token sampling.

**Why `response_format={"type": "json_object"}`**: Forces the model to output syntactically valid JSON. Combined with the prompt instruction, this gives two layers of format enforcement.

**The truncation strategy** (`_smart_truncate` at 8000–12000 chars):
- Keep more from the *end* of documents
- Why: Legal rulings ("THEREFORE, judgment is entered...") appear at the end. The beginning has boilerplate party information that's less useful for features.

#### `labels.py` — Supervision Pipeline (Separate from Feature Extraction)

**Why separate from `extraction.py`**:
- Features are the *inputs* to the model (X)
- Labels are the *targets* for training (y)
- They come from different documents (feature extraction reads all case docs; labels come only from judgment/order documents)
- They might use different models or prompts

Conflating them would create a training data pipeline that's fragile and hard to reason about.

---

### 4. Models — `models/`

**What**: Two gradient boosting models trained on extracted features + labels.

| Model | Task | Output |
|---|---|---|
| `ClassifierTrainer` | Binary classification | Win probability (0.0–1.0) |
| `RegressorTrainer` | Regression | Expected dollar amount |

#### Why Gradient Boosting (not logistic regression, not neural nets)?

**Gradient Boosting (`GradientBoostingClassifier`)** is an ensemble of decision trees trained sequentially, each correcting the errors of the previous one.

**Strengths for this use case**:
- Handles mixed feature types (numeric ratings 1–5, booleans, counts) natively
- Robust to outliers and noisy labels (court data is messy)
- Built-in feature importances for counterfactual analysis
- No need for feature scaling or one-hot encoding
- Strong performance with small-to-medium datasets (thousands of cases)

**Why not logistic regression**: Too simple for this feature space. Non-linear interactions (e.g., "evidence strength matters more when defendant doesn't appear") are hard to model linearly.

**Why not neural nets**: Would need tens of thousands of samples to generalize. Court data is limited (~thousands of cases). Gradient boosting extracts more signal from small datasets.

#### `tracking.py` — MLflow Integration

```python
with mlflow.start_run():
    mlflow.log_params({"n_estimators": 200, "max_depth": 5})
    mlflow.log_metrics({"roc_auc": roc_auc, "f1": f1})
    mlflow.sklearn.log_model(clf, "classifier")
    mlflow.register_model(model_uri, "litigation-win-classifier")
```

**What MLflow does**: Tracks every training run — parameters, metrics, the model artifact itself — and stores them in a registry. You can compare runs, roll back to an earlier version, and promote a model to "Production" to trigger API reload.

**The model registry stages**: `None → Staging → Production`. The API only loads models in `Production`. This creates a gate: you can train 10 models, evaluate them in Staging, and only the one you promote reaches users.

**Why not just save the model to disk**: If you save to disk, you lose the training parameters, metrics, and history. Six months later you can't tell which model you trained or why. MLflow gives you a complete audit trail.

---

### 5. API — `api/`

**What**: FastAPI service with 4 endpoints.

```
POST /predict          → single case prediction
POST /predict/batch    → up to 50 cases at once
POST /similar          → find similar historical cases
POST /counterfactual   → "what if" feature changes
```

#### Lifespan Context Manager

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    state.load_models()          # fetch Production models from MLflow
    state.load_feature_extractor()
    state.load_case_index()      # load FAISS index
    yield
    # cleanup on shutdown
```

**Why lifespan (not `@app.on_event("startup")`)**: `on_event` is deprecated in modern FastAPI. The lifespan context manager is the idiomatic replacement — it gives you a proper try/finally for cleanup on shutdown.

**Why load models at startup (not per request)**: Models are large (hundreds of MB). Loading from MLflow on every request would add seconds of latency. Load once at startup, keep in memory, serve instantly.

#### `/predict` Flow

1. `PredictionRequest` arrives (case text + optional metadata)
2. `FeatureExtractor.extract()` — LLM converts text to `FeatureVector`
3. `FeatureVector.to_model_input()` — converts to flat dict of floats
4. `classifier.predict_proba()` → win probability
5. `regressor.predict()` → expected dollar amount
6. Confidence computed from prediction and feature completeness
7. Returns `PredictionResponse`

**Why async LLM calls**: Feature extraction calls an external LLM API (network I/O). `async/await` lets FastAPI handle other requests while waiting for the LLM response, instead of blocking a thread. This is the core reason to use async in web APIs.

---

### 6. Retrieval — `retrieval/`

**The problem**: A win probability number isn't actionable. Users need to understand *why* and see real examples.

**The solution**: Find historical cases similar to the user's case and show their outcomes.

#### `embeddings.py` — Sentence Transformers

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode(text, normalize_embeddings=True)
```

**What**: Converts case text into a 384-dimensional vector (a point in high-dimensional space) where semantically similar cases are close together.

**`all-MiniLM-L6-v2`** is a small (22M parameter) but effective model trained specifically for semantic similarity. It runs on CPU in milliseconds per query.

**Why `normalize_embeddings=True`**: L2 normalization makes all vectors unit length, so **cosine similarity = dot product**. This is important because FAISS's `IndexFlatIP` (inner product) computes cosine similarity after normalization.

#### `index.py` — FAISS Search

```python
index = faiss.IndexFlatIP(embedding_dim)  # exact search, inner product
index.add(embeddings)                      # add all historical case embeddings
scores, indices = index.search(query_embedding, top_k)
```

**What**: FAISS (Facebook AI Similarity Search) stores all historical case embeddings and finds the nearest neighbors to a query in milliseconds — even across millions of vectors.

**Why FAISS over a database**: SQL `WHERE` clauses can't do vector similarity. FAISS is specifically built for this. For small indexes (thousands of cases), `IndexFlatIP` (exact search) is fine. For millions, you'd switch to `IndexIVFFlat` (approximate search with partitioning).

**The similarity threshold (0.3)**: Filters out poor matches. Below 0.3 cosine similarity, cases aren't semantically similar enough to be meaningful. Without this, the search always returns something even when nothing is genuinely similar.

---

### 7. Counterfactual — `counterfactual/`

**The problem**: "Your win probability is 42%" is not actionable. "If you had a written contract, your win probability would be 61%" is.

**The solution**: Feature perturbation analysis.

#### `analyzer.py` — How Counterfactuals Work

```python
def analyze(self, feature_vector, perturbations=None):
    baseline = classifier.predict_proba(feature_vector)
    for feature, new_value in perturbations.items():
        perturbed = feature_vector.copy()
        perturbed[feature] = new_value
        new_prob = classifier.predict_proba(perturbed)
        delta = new_prob - baseline
        results.append(CounterfactualResult(feature, delta, ...))
    return sorted(results, key=lambda r: r.win_prob_delta, reverse=True)
```

**What**: Takes a feature vector, perturbs each feature one at a time, re-runs the model, and reports the change in win probability.

**Why sorted by delta (descending)**: The user sees the changes that would help them most first. If getting a written contract adds 18% and adding a witness adds 4%, the contract advice comes first.

**The `FEATURE_CONSTRAINTS` guard**:
```python
evidence_strength: (1, 5)
contract_present: (0, 1)
```
Perturbations are clamped to valid ranges. You can't "increase evidence strength to 7" — that's outside the model's training distribution and would produce unreliable predictions.

**Why this is better than LIME/SHAP for this use case**: LIME and SHAP explain the prediction ("feature X contributed +Y to the score"). Counterfactuals explain the *action* ("if you change X, your score goes up by Y"). For a self-represented litigant, the second is far more useful.

---

### 8. Infrastructure

#### Docker — Two Images

**`Dockerfile.inference`**: Lightweight. Only needs sklearn (no GPU), FastAPI, FAISS. Runs `uvicorn api.app:app`.

**`Dockerfile.features`**: Heavier. Needs the OpenAI SDK, sentence-transformers. Could be separated as its own service for independent scaling (LLM calls are the bottleneck, not predictions).

**Multi-stage builds**:
```dockerfile
FROM python:3.11 AS builder
RUN pip install ...
COPY src/ .

FROM python:3.11-slim AS runtime
COPY --from=builder /usr/local/lib/python3.11/site-packages/ ...
```

**Why multi-stage**: The builder stage installs dependencies (large). The runtime stage only copies what's needed. This reduces the final image size significantly — no build tools, no compiler, just the installed packages.

#### Terraform — AWS IaC

Key resources in `infra/`:
- **S3 buckets**: `data/raw/` + `data/processed/` mirror on S3; `mlflow-artifacts/` for model storage
- **ECS Fargate**: Runs Docker containers without managing EC2 instances
- **IAM roles**: Principle of least privilege — each service only has access to what it needs

**Why Terraform over CloudFormation**: Terraform is cloud-agnostic (could switch to GCP by changing the provider), has better state management, and a larger community ecosystem.

#### GitHub Actions — CI/CD

**On every PR** (`ci.yml`):
1. `ruff check .` — lint (catches obvious bugs, style issues)
2. `ruff format --check .` — formatting
3. `mypy` — type checking (catches type errors before runtime)
4. `pytest tests/unit/` — unit tests

**On merge to main** (`docker-build.yml` + `deploy.yml`):
1. Build Docker images
2. Push to ECR
3. Update ECS task definitions
4. Restart ECS services
5. Run integration tests against deployed service

**Why mypy in CI**: Python is dynamically typed but mypy lets you opt in to static type checking. Combined with Pydantic's runtime validation, you get two layers of type safety: mypy catches bugs at development time, Pydantic catches them at runtime.

---

## Key Patterns Used Across the Project

### 1. Content-Addressable Caching

Used in both `features/extraction.py` and `features/labels.py`.

```python
def cache_key(case_number, text, version):
    return sha256(f"{version}:{case_number}:{text}".encode()).hexdigest()[:16]
```

**Pattern**: Derive a unique cache key from the *content* (not just the identifier). Cache invalidation is automatic — change the input, get a new key, re-compute. Same idea as Git's object store and Docker layers.

### 2. Pydantic at Every Boundary

Data enters each module as a typed Pydantic model and leaves as a typed Pydantic model.

**Why**: Type errors are caught at the boundary, not deep inside business logic. "Fail fast" — better to get a `ValidationError` at ingestion than a `KeyError` three steps later.

### 3. LLM → Structured Data → Traditional ML

The "hybrid" architecture:

```
LLM (reading comprehension) → structured features → sklearn (prediction)
```

**Why not end-to-end LLM for prediction**: LLMs aren't calibrated. They can't give you "42% win probability" with meaningful confidence. Sklearn gradient boosting is calibrated, interpretable, and fast at inference time.

### 4. Idempotent Pipeline Stages

Every stage can be safely re-run:
- Scraper: manifest prevents re-scraping done dates
- Feature extraction: cache prevents re-calling LLM on same text
- Model training: MLflow tracks each run separately; re-training creates a new run, doesn't overwrite old ones
- Index building: FAISS index is rebuilt from scratch (idempotent)

**Why this matters**: Pipelines fail. Network timeouts, API rate limits, power outages. Idempotency means you can always safely retry without corrupting your data.

### 5. Sentinel Values for Missing Data

```python
null → -1.0  (features out of normal range 0–5)
```

Instead of dropping rows with missing features or imputing mean values, the model learns to handle `-1.0` as "unknown." This is more honest and lets the model potentially learn that "unknown contract status" predicts differently from "contract absent."

---

## What to Study Next

**If you want to go deeper on any component**:

- **LLM feature extraction quality**: Read about prompt engineering for information extraction. The quality of your features determines model quality — garbage in, garbage out. Key search: "LLM structured extraction reliability."

- **Gradient Boosting**: Understand how XGBoost and LightGBM improve on sklearn's GBM. For real training, you'd likely switch to one of these. [XGBoost paper](https://arxiv.org/abs/1603.02754) is readable.

- **FAISS and vector search**: The [FAISS wiki](https://github.com/facebookresearch/faiss/wiki) covers all index types. For scale, `IndexIVFFlat` (inverted file) or `HNSW` (graph-based approximate search) replace `IndexFlatIP`.

- **MLflow in production**: [MLflow docs on model registry](https://mlflow.org/docs/latest/model-registry.html). Key concept: lifecycle management (None → Staging → Production → Archived).

- **Counterfactual explanations**: LIME and SHAP are alternatives. [SHAP docs](https://shap.readthedocs.io/en/latest/) are excellent. The trade-off: SHAP is theoretically grounded (Shapley values from game theory) but harder to explain to users than simple "change X → win probability goes up Y%."

- **FastAPI production patterns**: [FastAPI docs on lifespan](https://fastapi.tiangolo.com/advanced/events/), background tasks, dependency injection.

**Related files to read**:
- [features/extraction.py](../features/extraction.py) — full feature extraction pipeline
- [models/trainer.py](../models/trainer.py) — GBM training + MLflow logging
- [retrieval/index.py](../retrieval/index.py) — FAISS index build + search
- [counterfactual/analyzer.py](../counterfactual/analyzer.py) — perturbation analysis
- [api/app.py](../api/app.py) — FastAPI endpoints + lifespan
