output "data_bucket_name" {
  description = "S3 bucket for case data"
  value       = aws_s3_bucket.data.bucket
}

output "data_bucket_arn" {
  value = aws_s3_bucket.data.arn
}

output "mlflow_bucket_name" {
  description = "S3 bucket for MLflow artifacts"
  value       = aws_s3_bucket.mlflow_artifacts.bucket
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "inference_service_name" {
  description = "ECS inference service name"
  value       = aws_ecs_service.inference.name
}
