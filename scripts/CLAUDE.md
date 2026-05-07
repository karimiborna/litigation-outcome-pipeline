  # Scripts

One-off CLI scripts for the data → training → deployment pipeline. Run from the repo root with `conda activate ML`.

## Pipeline Order

```
extract_features_from_local_cases.py   (1) LLM feature extraction → features_cache/
extract_missing_labels.py              (2) LLM label extraction   → data/processed/labels.json
build_training_rows.py                 (3) Join features + labels  → dataset.csv
train_models.py                        (4) Train XGBoost models    → MLflow registry
promote_models_to_production.py        (5) Promote to Production   → API picks them up
```

## Scripts

### `extract_features_from_local_cases.py`
Reads raw case text from `scraper/data/processed/*.txt`, builds `ProcessedCase` objects (excluding outcome docs to prevent leakage), and runs `FeatureExtractor.extract_batch` to write per-case JSON files to `data/features_cache/`. Existing cache entries are skipped — only new cases trigger LLM calls.

```bash
python scripts/extract_features_from_local_cases.py
```

### `extract_missing_labels.py`
Reads case text and extracts outcome labels (win/loss, amounts awarded, judgment date) using the LLM label pipeline. Writes/updates `data/processed/labels.json`. Existing labels are preserved — only unlabeled cases are sent through the LLM.

```bash
python scripts/extract_missing_labels.py
```

### `build_training_rows.py`
Joins `data/features_cache/*.json` with `data/processed/labels.json` by case number. Outputs `data/processed/training_rows.jsonl` and `dataset.csv` at the repo root. Also prints a coverage report showing how many cases have both features and labels.

```bash
python scripts/build_training_rows.py
```

### `train_models.py`
**Primary training path.** Loads `dataset.csv`, trains both models using the v2 `feat_*` feature set (51 model features after one-hot encoding), logs metrics and artifacts to MLflow, and registers both models in the registry.

- Classifier: `litigation-win-classifier` (XGBClassifier, binary win/loss)
- Regressor: `litigation-monetary-regressor` (XGBRegressor, dollar amount)

```bash
python scripts/train_models.py
```

### `promote_models_to_production.py`
Transitions the latest registered version of each model from `None`/`Staging` to `Production`. The API loads models at `models:/.../Production` on startup — run this after training to make new models live.

```bash
python scripts/promote_models_to_production.py
```

### `run_server.sh`
Starts the FastAPI server with auto-reload. Looks for `.venv` or `venv`; if using conda, run uvicorn directly instead:

```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

## Notes

- `train_models.py` requires `dataset.csv` at the repo root — run `build_training_rows.py` first if it's missing or stale.
- Feature extraction (`extract_features_from_local_cases.py`) uses `LLM_API_KEY` from `.env` — costs ~$0.01–0.05 per case.
- All model artifacts are tracked in MLflow at `http://35.208.251.175:5000` — never committed to git.
- XGBoost handles NaN natively so nullable features are passed through as-is (no imputation or row dropping).
