# Models Module

Trains and serves two scikit-learn models with MLflow tracking.

## File Map

| File | Purpose |
|---|---|
| `trainer.py` | ClassifierTrainer, RegressorTrainer, vectors_to_dataframe() |
| `tracking.py` | MLflow helpers — init, experiments, runs, logging, registry |
| `config.py` | MLflowConfig — tracking_uri, artifact_root, experiment/model names |

## Two Models

**Classifier** — plaintiff win/loss probability
- Algorithm: GradientBoostingClassifier (200 trees, depth=5, lr=0.1)
- Output: win probability (0–1) + confidence level
- Metrics: accuracy, precision, recall, F1, ROC-AUC

**Regressor** — expected monetary outcome
- Algorithm: GradientBoostingRegressor
- Output: expected dollar amount
- Metrics: MAE, RMSE, R²

## MLflow

All training runs are logged to MLflow. Models graduate through stages:
`None → Staging → Production`

Only Production-stage models are loaded by the API.

```bash
# Start MLflow server
mlflow server --config mlflow/server_config.yaml

# UI at http://localhost:5000
```

Model names in registry:
- `litigation-win-classifier`
- `litigation-monetary-regressor`

## Training

```python
from models.trainer import ClassifierTrainer, RegressorTrainer
from models.tracking import init_mlflow

init_mlflow()
clf = ClassifierTrainer()
clf.train(feature_vectors, labels)  # logs to MLflow automatically
```

## Key Rule

Model artifacts are **never committed to git** — always loaded from MLflow registry.
