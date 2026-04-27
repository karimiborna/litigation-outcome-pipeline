# Docker Module

Containerization configuration for reproducible builds and deployments.

## Responsibilities

- Dockerfile(s) for each service:
  - Feature extraction service (LLM-dependent, heavier)
  - Model inference / API service (lightweight, fast)
- Docker Compose for local multi-service development
- Base image selection and dependency management
- Build optimization (layer caching, multi-stage builds)

## Key Considerations

- Separate containers for feature extraction and inference to allow independent scaling
- Keep images small — use multi-stage builds and slim base images
- Pin dependency versions for reproducibility
- Secrets (API keys for LLM, cloud credentials) must NOT be baked into images — use env vars or secret managers
- **Docker Compose does not run a local MLflow container** — both services point at the hosted GCP server (`http://35.208.251.175:5000`) via `MLFLOW_TRACKING_URI` in `docker-compose.yml`
- Images are built and pushed to GHCR by GitHub Actions (see `.github/workflows/docker-build.yml`)

## Inference image (`Dockerfile.inference`)

- Builds with **`pip install .`** from repo **`pyproject.toml`** so all package dependencies match the project (not a hand-picked subset).
- **CI image names:** `docker/metadata-action` sets `images:` to `ghcr.io/${{ github.repository }}-${{ matrix.service }}` — e.g. **`ghcr.io/<owner>/<repo>-inference`** for the API (see workflow matrix `service: inference`). README uses placeholder **`litigation-inference`** for manual `docker build -t` examples.
- **Runtime:** container must reach **`MLFLOW_TRACKING_URI`** (not `localhost` unless MLflow is in the same Docker network). **`LLM_API_KEY`** required for `/predict` as implemented.
