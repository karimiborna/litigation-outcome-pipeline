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
- Docker Compose should replicate the production topology locally
- Images are built and pushed by GitHub Actions (see `.github/workflows/`)
