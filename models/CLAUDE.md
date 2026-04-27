# Models Module

Classification and regression model training for predicting case outcomes.

## Responsibilities

| File | Purpose |
|---|---|
| `trainer.py` | ClassifierTrainer, RegressorTrainer |
| `tracking.py` | MLflow helpers — init, experiments, runs, logging, registry |
| `config.py` | MLflowConfig — tracking_uri, artifact_root, experiment/model names |
| `dataset.py` | Shared preprocessing — `preprocess_feature_frame()`, `feature_vector_to_model_frame()`, `MODEL_FEATURE_COLUMNS` |

## Key Considerations

- Two distinct prediction targets: binary outcome (win/lose) and continuous outcome (monetary amount)
- Models should be compared across experiments using MLflow tracking
- Feature importance / interpretability is important for downstream explanation generation
- Train/validation/test splits must be consistent and reproducible
- Model artifacts are never committed to git — always tracked via MLflow

## Scripts and registry workflow

- **`scripts/train_classifier_real.py`** — **primary training path**. Trains classifier + regressor on **real `dataset.csv`** using the `v2 feat_*` preprocessing in `models.dataset`. Logs dataset SHA-256, feature columns artifact, and metrics. Registers `litigation-win-classifier` and `litigation-monetary-regressor`.
- **`scripts/train_binary_classifier.py`** — trains the same two models on **synthetic** data (for demo/smoke test purposes). Uses the older `FeatureVector.to_model_input()` columns, not the v2 feature set.
- **`scripts/promote_models_to_production.py`** — moves latest registered version of each model to **Production** (API loads `models:/.../Production`). MLflow may warn that stages are deprecated in favor of aliases in a future major version.
- **`models/tracking.get_or_create_experiment`** — for **remote HTTP** tracking URIs, does **not** pass a client `artifact_location` (server must use **`--serve-artifacts`**). If an existing experiment still has a **server-local** `artifact_location` (`/home/...` or `file:`), training uses a sibling experiment name with suffix **`-remote-artifacts`** so laptops can upload artifacts.

## Feature columns (v2)

`dataset.py` defines `MODEL_FEATURE_COLUMNS` — the 35 columns the model actually trains on. These are `feat_*` prefixed columns from `dataset.csv`. The API uses `feature_vector_to_model_frame()` to convert a `FeatureVector` into this same column layout at inference time, ensuring training/serving parity.
