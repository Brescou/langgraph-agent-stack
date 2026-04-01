terraform {
  required_version = ">= 1.6"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.25"
    }
  }
}

# WARNING: No backend is configured — Terraform will store state LOCALLY.
# Local state is unsuitable for team or production use because:
#   * State files may contain secrets (API keys, passwords)
#   * No locking — concurrent applies can corrupt state
#   * No history or audit trail
#
# REQUIRED for production: uncomment ONE of the backend blocks below.
#
# For GCP:
#   terraform {
#     backend "gcs" {
#       bucket = "your-terraform-state-bucket"
#       prefix = "langgraph-agent-stack"
#     }
#   }
#
# For AWS:
#   terraform {
#     backend "s3" {
#       bucket         = "your-terraform-state-bucket"
#       key            = "langgraph-agent-stack/terraform.tfstate"
#       region         = "us-east-1"
#       dynamodb_table = "terraform-locks"
#       encrypt        = true
#     }
#   }
