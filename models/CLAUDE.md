# Models Module

Trains and serves two scikit-learn models with MLflow tracking.

## File Map

| File | Purpose |
|---|---|
| `trainer.py` | ClassifierTrainer, RegressorTrainer, vectors_to_dataframe() |
| `tracking.py` | MLflow helpers — init, experiments, runs, logging, registry |
| `config.py` | MLflowConfig — tracking_uri, artifact_root, experiment/model names |

## Two Models

- Two distinct prediction targets: binary outcome (win/lose) and continuous outcome (monetary amount)
- Models should be compared across experiments using MLflow tracking
- Feature importance / interpretability is important for downstream explanation generation
- Train/validation/test splits must be consistent and reproducible
- Model artifacts are never committed to git — always tracked via MLflow

## Scripts and registry workflow

- **`scripts/train_binary_classifier.py`** — trains classifier + regressor on **synthetic** data (same feature columns as `FeatureVector.to_model_input()`), logs metrics, registers **`litigation-win-classifier`** and **`litigation-monetary-regressor`**. Requires reachable MLflow with model registry.
- **`scripts/promote_models_to_production.py`** — moves latest registered version of each model to **Production** (API loads `models:/.../Production`). MLflow may warn that stages are deprecated in favor of aliases in a future major version.
- **`models/tracking.get_or_create_experiment`** — for **remote HTTP** tracking URIs, does **not** pass a client `artifact_location` (server must use **`--serve-artifacts`**). If an existing experiment still has a **server-local** `artifact_location` (`/home/...` or `file:`), training uses a sibling experiment name with suffix **`-remote-artifacts`** so laptops can upload artifacts.

## Still to do for real models

- Replace synthetic training with **real feature rows + labels** from processed cases (`features/` + `features/labels.py` or manual labels), then retrain and register new versions.
