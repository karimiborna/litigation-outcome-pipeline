# MLflow Module

Experiment tracking, metric logging, model versioning, and model registry configuration.

## Responsibilities

- Configure MLflow tracking server (local or remote)
- Define experiment structure and naming conventions
- Log parameters, metrics, and artifacts during training runs
- Register trained models in the MLflow model registry with stage transitions (Staging → Production)
- Serve models for inference via MLflow's serving capabilities

## Key Considerations

- All model training must go through MLflow — no untracked experiments
- Artifact storage should point to cloud storage (S3/GCS) in production
- Model registry stages: None → Staging → Production
- Tracking URI and artifact location are environment-dependent (local dev vs cloud)
- MLflow configuration files and server setup live here; actual model code lives in `models/`
