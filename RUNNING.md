# Running the Litigation Outcome Pipeline

Practical guide for running the API + frontend locally and (re)training the models.

## Prerequisites

- conda env `ML` activated (do **not** use `.venv` / `venv` — the project standardizes on conda)
- Network access to the hosted MLflow server: http://35.208.251.175:5000
- A populated `.env` at the repo root (the existing one in this repo is fine; only `LLM_API_KEY` matters for inference)

```bash
conda activate ML
cd ~/Desktop/MLOPS-Project/litigation-outcome-pipeline
```

All commands below assume that working directory and a `(ML)` shell.

---

## Run the API + Frontend

The API and the LexRatio frontend are served by the same FastAPI process. There is no separate React/Next dev server. `lexratio.html` is a static file that FastAPI serves at `GET /`.

### One command

```bash
./scripts/run_server.sh
```

Then open in a browser:

| URL | What it is |
|---|---|
| http://localhost:8000 | LexRatio frontend (the form) |
| http://localhost:8000/lexratio | Same frontend, alternate route |
| http://localhost:8000/docs | Swagger / OpenAPI docs |
| http://localhost:8000/health | JSON health check |

`run_server.sh` handles everything that bites otherwise:

- Sets the macOS fork-safety env vars (`OBJC_DISABLE_INITIALIZE_FORK_SAFETY`, `no_proxy=*`, `KMP_DUPLICATE_LIB_OK`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) **before** Python starts. Without these, uvicorn segfaults on macOS during model load.
- Defaults `MLFLOW_TRACKING_URI` to the hosted GCP server (no localhost fallback).
- Runs uvicorn with `--reload` for development. Set `ENV=production ./scripts/run_server.sh` to disable reload.

### Verify it's alive

```bash
curl -s http://localhost:8000/health | jq .
```

Should return:
```json
{"status":"healthy","version":"0.1.0","models_loaded":true,"classifier_loaded":true,"regressor_loaded":true}
```

If `models_loaded: false`, the registry has no Production version — see [Train and promote models](#train-and-promote-models).

### Test the prediction endpoint

```bash
curl -s -X POST http://localhost:8000/api/analyze-lexratio \
  -H "Content-Type: application/json" \
  -d '{
    "case_text": "I am suing one defendant, a contractor not represented by an attorney. We signed a written contract on March 3, 2026 for $1,500 to repair my roof. I paid in full on March 4. The contractor stopped showing up after one day. I have receipts, photos, text messages, and a witness. I sent a certified-mail demand letter on March 20 with no response. Defendant has not filed any response.",
    "claim_amount": 1500,
    "cause_of_action": "contract"
  }' | jq .
```

---

## Train and promote models

Only needed if MLflow has no Production version of the models, or you want to retrain on fresh data.

### Pipeline order

```
extract_features_from_local_cases.py   (1) LLM feature extraction → data/features_cache/
extract_missing_labels.py              (2) LLM label extraction   → data/processed/labels.json
build_training_rows.py                 (3) Join features + labels  → dataset.csv
train_models.py                        (4) Train XGBoost models    → MLflow registry
promote_models_to_production.py        (5) Promote to Production   → API picks them up
```

Steps 1-3 are slow (LLM calls) and only need to run when new cases come in. `dataset.csv` is committed to the repo at the moment, so you can usually skip straight to step 4.

### Steps 4 + 5: train and promote

```bash
python scripts/train_models.py
python scripts/promote_models_to_production.py
```

`train_models.py` reads `dataset.csv`, trains an XGBoost classifier (`litigation-win-classifier`) and regressor (`litigation-monetary-regressor`) on the v2 `feat_*` feature set, logs to MLflow, and registers new versions. `promote_models_to_production.py` flips the latest registered version of each to Production stage so the API picks it up on next startup.

After running, verify in the browser: http://35.208.251.175:5000/#/models — both models should have a version stamped Production.

Restart the API to pick up the new models:

```bash
# Ctrl+C in the run_server.sh terminal, then:
./scripts/run_server.sh
```

### Steps 1-3: full data refresh (optional)

```bash
python scripts/extract_features_from_local_cases.py   # ~$0.01-0.05 per case via LLM
python scripts/extract_missing_labels.py              # similar cost
python scripts/build_training_rows.py                 # writes dataset.csv
```

Costs `LLM_API_KEY` budget. Existing cache entries are skipped — only new cases trigger LLM calls.

---

## How the pieces fit together

```
Browser  ──>  GET / (lexratio.html)
             │
             └──>  POST /api/analyze-lexratio
                       │
                       ├──>  FeatureExtractor (LLM, OpenAI gpt-4o-mini)
                       │         → FeatureVector (~40 boolean/int features)
                       │
                       └──>  feature_vector_to_model_frame
                                 │
                                 ├──>  classifier.predict_proba   (XGBoost, MLflow Production)
                                 └──>  regressor.predict          (XGBoost, MLflow Production)
                                       │
                                       └──>  LexRatioAnalysisResponse
```

Models are loaded once at startup from `models:/<name>/Production` on the hosted MLflow server. The retrieval / case-index endpoint is currently unwired (no `data/retrieval_index/` on disk) — `/similar` will return empty until the index is built.

---

## Troubleshooting

### `zsh: segmentation fault` during startup

Use `./scripts/run_server.sh`. Launching uvicorn directly without the env vars from that script reliably segfaults on macOS because XGBoost loads `libomp` and MLflow's artifact downloader then forks. The script sets the necessary fork-safety env vars in the shell **before** Python starts, which is the only place they take effect in time.

### `models_loaded: false` on `/health`

The registry has no Production version of one or both models.

```bash
python scripts/promote_models_to_production.py
```

If that fails because there are no registered versions at all, run `python scripts/train_models.py` first.

### `422 Unprocessable Entity` on POST `/api/analyze-lexratio`

Two flavors:

1. **`case_text` too short** — the schema requires `case_text >= 10` chars. Fill in the narrative properly.
2. **`Missing required inference features: [...]`** — the LLM left certain fields null. Re-submit with explicit mentions in the case text, e.g. "I am suing one defendant" and "the opposing party is/is not represented by an attorney." (This is a known sharp edge — XGBoost can handle NaN natively, but the inference-time validator currently rejects nulls.)

### `.env` parse error when `source`-ing it

Don't `source .env`. The Python configs (`MLflowConfig`, `FeaturesConfig`, `RetrievalConfig`) read it directly via pydantic-settings, and `run_server.sh` doesn't source it either. Shell-sourcing breaks because line 13 contains unquoted parens (`SCRAPER_USER_AGENT=... (SF Small Claims academic study)`).

### MLflow tracking URI keeps defaulting to localhost

You're not using `run_server.sh`. The script forces `MLFLOW_TRACKING_URI=http://35.208.251.175:5000` if unset. If you're launching uvicorn directly:

```bash
export MLFLOW_TRACKING_URI=http://35.208.251.175:5000
export MLFLOW_REGISTRY_URI=http://35.208.251.175:5000
```

---

## Useful URLs

- LexRatio frontend: http://localhost:8000
- API docs (Swagger): http://localhost:8000/docs
- Health check: http://localhost:8000/health
- MLflow UI: http://35.208.251.175:5000
- MLflow model registry: http://35.208.251.175:5000/#/models
