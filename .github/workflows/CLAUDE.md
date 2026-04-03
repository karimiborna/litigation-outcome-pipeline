# CI/CD Workflows

Four GitHub Actions workflows automate testing, validation, and deployment.

## Workflows

**`ci.yml`** — Runs on every PR
- Lint with ruff
- Type check with mypy
- Run all unit tests (pytest)
- Fails PR if any step fails

**`data-validation.yml`** — Runs on push to main
- Validates data schemas
- Checks that raw/processed data structure is intact

**`docker-build.yml`** — Runs on push to main
- Builds both Docker images (features + inference)
- Pushes to ECR

**`deploy.yml`** — Runs on push to main (after docker-build)
- Deploys updated images to ECS
- Runs integration tests against deployed service

## Secrets Required

Set in GitHub repo settings → Secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `NVIDIA_API_KEY`
- `MLFLOW_TRACKING_URI`

## Local CI Check

```bash
ruff check .
mypy .
pytest tests/unit/
```
