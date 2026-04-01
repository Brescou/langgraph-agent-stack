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

# To enable remote state (REQUIRED for team/production use), add a backend block:
# 
# For GCP:
#   backend "gcs" {
#     bucket = "your-terraform-state-bucket"
#     prefix = "langgraph-agent-stack"
#   }
#
# For AWS:
#   backend "s3" {
#     bucket         = "your-terraform-state-bucket"
#     key            = "langgraph-agent-stack/terraform.tfstate"
#     region         = "us-east-1"
#     dynamodb_table = "terraform-locks"
#   }
