# CI/CD Workflows

GitHub Actions workflows for automated testing, validation, building, and deployment.

## Responsibilities

**`ci.yml`** — Runs on every PR and push to main
- Lint with ruff (`ruff check`)
- Format check with ruff (`ruff format --check`)
- Type check with mypy (scoped to `scraper/` and `data/`)
- Run all unit tests with pytest (Python 3.10 + 3.12 matrix)
- Fails PR if any step fails

## Key Considerations

**`docker-build.yml`** — Runs on push to main
- Builds both Docker images (features + inference)
- Pushes to **GitHub Container Registry (GHCR)** — not ECR
- Images tagged as `ghcr.io/<owner>/<repo>-inference` and `ghcr.io/<owner>/<repo>-features`
- Uses `GITHUB_TOKEN` for auth — no extra secrets needed for build/push

**`deploy.yml`** — Runs on push to main (after docker-build)
- Deploys updated images to ECS
- Runs integration tests against deployed service

## Secrets Required

Set in GitHub repo settings → Secrets:
- `NVIDIA_API_KEY`
- `MLFLOW_TRACKING_URI` (should be `http://35.208.251.175:5000`)
- AWS credentials only needed if/when ECS deploy is wired up

## Local CI Check

```bash
ruff check .
ruff format --check .
mypy scraper/ data/ --ignore-missing-imports
pytest tests/unit/
```
