# ---------------------------------------------------------------------------
# This root directory is NOT a deployable Terraform module.
#
# Each cloud provider has its own entry-point directory with its own
# versions.tf that pins the required providers:
#
#   infra/terraform/gke/   — google  ~> 7.0, helm ~> 3.1, kubernetes ~> 3.0
#   infra/terraform/eks/   — aws     ~> 6.0, helm ~> 3.1, kubernetes ~> 3.0
#   infra/terraform/aks/   — azurerm ~> 4.0, helm ~> 3.1, kubernetes ~> 3.0
#
# See main.tf for deployment instructions.
# ---------------------------------------------------------------------------

terraform {
  required_version = ">= 1.6"
}

# WARNING: No backend is configured — Terraform will store state LOCALLY.
# Local state is unsuitable for team or production use because:
#   * State files may contain secrets (API keys, passwords)
#   * No locking — concurrent applies can corrupt state
#   * No history or audit trail
#
# REQUIRED for production: configure a remote backend in the cloud-specific
# entry-point directory (gke/, eks/, or aks/).
#
# For GCP (GCS):
#   terraform {
#     backend "gcs" {
#       bucket = "your-terraform-state-bucket"
#       prefix = "langgraph-agent-stack"
#     }
#   }
#
# For AWS (S3):
#   terraform {
#     backend "s3" {
#       bucket         = "your-terraform-state-bucket"
#       key            = "langgraph-agent-stack/terraform.tfstate"
#       region         = "us-east-1"
#       dynamodb_table = "terraform-locks"
#       encrypt        = true
#     }
#   }
#
# For Azure (Blob Storage):
#   terraform {
#     backend "azurerm" {
#       resource_group_name  = "your-tfstate-rg"
#       storage_account_name = "yourtfstatesa"
#       container_name       = "tfstate"
#       key                  = "langgraph-agent-stack.tfstate"
#     }
#   }
