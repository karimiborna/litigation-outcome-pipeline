# Litigation Outcome Pipeline

An end-to-end machine learning system for predicting the outcomes of small claims court cases. The system combines structured metadata with unstructured legal text to produce two predictions: the probability of a plaintiff win and the expected monetary outcome.

Rather than using an LLM directly for prediction, the system uses one strictly for feature extraction — converting raw case text into structured signals like evidence strength, contract presence, and argument clarity. These features are merged with case metadata (claim amount, case type, etc.) and fed into traditional classification and regression models.

Case data is scraped from the SF Superior Court public records system and extracted from PDFs using the NVIDIA API. All model development is tracked with MLflow for experiment logging, metric comparison, and model versioning.

Beyond prediction, the system includes two interpretability modules:
- **Similar case retrieval** — uses embeddings to find historical cases with similar characteristics, grounding explanations in real examples
- **Counterfactual analysis** — simulates changes to key features and shows how those changes would shift the predicted outcome

The system is built with production-level MLOps practices: containerized with Docker, deployed to cloud infrastructure (AWS ECS / GCP Cloud Run), and automated with GitHub Actions for CI/CD.

## Pipeline

```
scraper → data → features → models → api
                               ↕
                            mlflow

         retrieval ←──── api ────→ counterfactual

         docker + infra + .github/workflows wrap the whole thing
```

**`scraper/`** — Hits the SF Superior Court website by date, downloads case PDFs, and sends them through the NVIDIA API for text extraction. This is the entry point and bottleneck (~500 extractions/day/key).

**`data/`** — Landing zone. Raw PDFs go into `raw/`, extracted text goes into `processed/`, and `schemas/` defines what valid data looks like. Nothing downstream touches raw — it's immutable.

**`features/`** — Takes the processed case text and runs it through an LLM to extract structured signals (evidence strength, contract presence, argument clarity, etc.). Merges those with metadata like claim amount and case type into a unified feature matrix.

**`models/`** — Consumes the feature matrix. Trains two models: a classifier (plaintiff win probability) and a regressor (expected monetary outcome). All training runs are logged to MLflow.

**`mlflow/`** — Sits alongside `models/` as the tracking backbone. Every experiment, metric, parameter, and artifact is versioned here. Models graduate through stages (Staging → Production) in the registry.

**`retrieval/`** — Embeds case text, builds a vector index, and finds the most similar historical cases. Used to generate explanations grounded in real examples.

**`counterfactual/`** — Takes a case's features, perturbs them, re-runs the models, and shows how the outcome shifts.

**`api/`** — The serving layer. Exposes endpoints for prediction, similar case retrieval, and counterfactual analysis. Loads models from MLflow and orchestrates the full inference path.

**`docker/`** — Wraps the API and feature extraction into containers. Separate images for the LLM-heavy feature service and the lightweight inference service.

**`infra/`** — Deploys containers to AWS ECS or GCP Cloud Run, provisions S3/GCS buckets for data and MLflow artifacts.

**`.github/workflows/`** — Automates testing on every PR, Docker image builds on merge, and deployment to cloud.

**`tests/`** — Unit and integration tests mirroring the source module structure.
