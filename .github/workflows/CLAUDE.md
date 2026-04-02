# CI/CD Workflows

GitHub Actions workflows for automated testing, validation, building, and deployment.

## Responsibilities

- **Test** — run unit and integration tests on each push/PR
- **Lint** — code quality checks (formatting, type checking)
- **Data validation** — verify data schemas and integrity on data changes
- **Docker build** — build and push container images on merge to main
- **Deploy** — deploy updated services to cloud infrastructure (ECS / Cloud Run)

## Key Considerations

- Workflows trigger on push to main and on pull requests
- Tests must pass before merge is allowed
- Docker images are tagged with git SHA for traceability
- Deployment workflows need cloud credentials stored as GitHub Secrets
- Separate workflows for CI (test/lint) and CD (build/deploy) for clarity
- Data validation workflow should run when anything in `data/schemas/` changes
