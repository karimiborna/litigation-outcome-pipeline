# Infra Module

Terraform configuration for AWS deployment.

## Structure

```
infra/
├── main.tf          # Provider config (AWS)
├── s3.tf            # S3 buckets — raw data, processed data, MLflow artifacts
├── ecs.tf           # ECS cluster + task definitions for features + api services
├── variables.tf     # Input variables
├── outputs.tf       # Output values (bucket names, service URLs)
└── envs/
    ├── dev.tfvars   # Dev environment values
    └── prod.tfvars  # Prod environment values
```

## Deploy

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

AWS (S3 + ECS) is the chosen cloud platform. GCP (GCS + Cloud Run) was the alternative.
