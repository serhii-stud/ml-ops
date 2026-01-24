provider "aws" {
  region = "us-east-1"
}

# 1. ECR Repositories (для ваших кастомных имиджей)
resource "aws_ecr_repository" "training_repo" {
  name = "banking-training-tf"
  image_tag_mutability = "MUTABLE"
}

resource "aws_ecr_repository" "monitoring_repo" {
  name = "banking-offline-monitor"
}

# 2. S3 Bucket (если он еще не управляется TF)
resource "aws_s3_bucket" "mlops_bucket" {
  bucket = "mlops-project-sm-tf"
}

# 3. IAM Roles (SageMaker Execution Role)
# Здесь нужно описать роль, которую вы используете: arn:aws:iam::191072691166:role/ml-ops-SageMaker-ExecutionRole
# Terraform позволит добавить к ней нужные права (например, доступ к новому ECR).