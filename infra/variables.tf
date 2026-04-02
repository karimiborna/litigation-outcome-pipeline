variable "aws_region" {
  description = "AWS region for all resources"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Deployment environment (dev, staging, production)"
  type        = string
  default     = "dev"
}

variable "project_name" {
  description = "Project name used for resource naming"
  type        = string
  default     = "litigation-pipeline"
}

# ECS
variable "inference_cpu" {
  description = "CPU units for the inference service (1024 = 1 vCPU)"
  type        = number
  default     = 512
}

variable "inference_memory" {
  description = "Memory (MiB) for the inference service"
  type        = number
  default     = 1024
}

variable "feature_cpu" {
  description = "CPU units for the feature extraction service"
  type        = number
  default     = 1024
}

variable "feature_memory" {
  description = "Memory (MiB) for the feature extraction service"
  type        = number
  default     = 2048
}

variable "desired_count" {
  description = "Desired number of ECS tasks"
  type        = number
  default     = 1
}

variable "ecr_repository_url" {
  description = "ECR repository URL for Docker images"
  type        = string
  default     = ""
}
