# Infrastructure Module

Cloud deployment configuration for AWS or GCP.

## Responsibilities

- Define cloud resources for:
  - **Storage** — S3 bucket or GCS bucket for raw/processed data and MLflow artifacts
  - **Compute** — ECS task definitions or Cloud Run service configs for API containers
  - **Networking** — load balancers, security groups, IAM roles as needed
- Infrastructure-as-code (Terraform, CloudFormation, or deployment scripts)
- Environment configuration (dev, staging, production)

## Key Considerations

- Target platform: AWS (S3 + ECS) or GCP (GCS + Cloud Run) — decide early
- Keep infra definitions declarative and version-controlled
- Separate configs per environment (dev/prod)
- MLflow tracking server may also need cloud hosting
- Auto-scaling policies for inference services
- Cost management — use spot/preemptible instances where possible for non-critical workloads
