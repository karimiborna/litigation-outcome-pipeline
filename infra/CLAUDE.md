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

```bash
cd infra/
terraform init
terraform apply -var-file=envs/prod.tfvars
```

## Resources Provisioned

- **S3**: data bucket (raw/ + processed/), MLflow artifacts bucket
- **ECS**: cluster, two task definitions (features service, inference API)
- **IAM**: task execution roles with S3 read access

## Platform Decision

AWS (S3 + ECS) is the target for **application serving** — the `inference` and `features` ECS services defined in `ecs.tf`.

MLflow tracking is **not** on AWS. The MLflow server is hosted on a **GCP VM** at `http://35.208.251.175:5000` — see `../mlflow/CLAUDE.md`. ECS tasks should set `MLFLOW_TRACKING_URI` to the GCP URL, not to an in-cluster service. (The current `ecs.tf:93` placeholder `http://mlflow:5000` is wrong and should be replaced before deploy.)

## Status

This Terraform is declared but **not yet applied**. No `.terraform/`, `tfstate`, or `tfvars` files exist on disk. Before `terraform apply`:

1. Create `envs/dev.tfvars` and `envs/prod.tfvars` (referenced in this doc but missing).
2. Populate `network_configuration.subnets` and `security_groups` in the ECS service block.
3. Update `MLFLOW_TRACKING_URI` in `ecs.tf` to point at the hosted GCP MLflow URL.
