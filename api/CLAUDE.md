# API Module

Inference service for serving predictions, retrievals, and counterfactual analyses.

## Responsibilities

- Expose REST endpoints for:
  - **Prediction** — accept case data, return win probability + expected monetary outcome
  - **Similar cases** — return top-K similar historical cases with explanations
  - **Counterfactual** — accept feature changes, return outcome deltas
- Load trained models from MLflow model registry
- Run the feature extraction pipeline on incoming raw case data
- Handle request validation, error responses, and health checks

## Key Considerations

- Separate services for feature extraction (LLM calls) and model inference (fast)
- Feature extraction has higher latency due to LLM calls — consider async or pre-extraction
- Models are loaded from MLflow — service needs access to the registry
- Must support both batch and real-time inference
- Input validation is critical — this is the system boundary
- Containerized via Docker (see `docker/`)
- Deployed to AWS ECS or Google Cloud Run (see `infra/`)
