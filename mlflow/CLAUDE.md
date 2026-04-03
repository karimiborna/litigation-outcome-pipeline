# MLflow Module

Experiment tracking and model registry for the pipeline.

## Server

```bash
mlflow server --config mlflow/server_config.yaml
# UI at http://localhost:5000
```

Config (`server_config.yaml`):
- Backend store: SQLite (`mlruns/mlflow.db`)
- Artifact root: `mlruns/artifacts`
- Host: `0.0.0.0:5000`

## Experiments

| Experiment | Models Tracked |
|---|---|
| `litigation-classifier` | Win/loss classifier runs |
| `litigation-regressor` | Monetary outcome regressor runs |

## Model Registry

Models graduate through stages:
```
None → Staging → Production
```

Only `Production` models are loaded by the API at startup.

Model names:
- `litigation-win-classifier`
- `litigation-monetary-regressor`

## Key Rule

Model artifacts are **never committed to git**. Everything lives in MLflow. The `mlruns/` directory is gitignored.

## Helper Functions (models/tracking.py)

- `init_mlflow()` — connect to tracking server
- `get_or_create_experiment()` — idempotent experiment setup
- `start_run()` — begin a tracked run
- `log_metrics()` — log dict of metrics
- `log_model_artifact()` — register sklearn model
- `transition_model_stage()` — move between None/Staging/Production
- `load_production_model()` — fetch model for inference
