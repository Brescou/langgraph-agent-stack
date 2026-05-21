output "cluster_endpoint" {
  description = "GKE cluster API server endpoint."
  value       = google_container_cluster.main.endpoint
}

output "cluster_name" {
  description = "Name of the GKE Autopilot cluster."
  value       = google_container_cluster.main.name
}

output "namespace" {
  description = "Kubernetes namespace where the langgraph-agent-stack is deployed."
  value       = module.gke.namespace
}

output "helm_release_status" {
  description = "Status of the langgraph Helm release."
  value       = module.gke.helm_release_status
}

output "gcp_service_account_email" {
  description = "GCP service account used by the workload via Workload Identity."
  value       = module.gke.gcp_service_account_email
}

output "anthropic_secret_id" {
  description = "Secret Manager secret ID for ANTHROPIC_API_KEY (populate outside Terraform)."
  value       = module.gke.anthropic_secret_id
}

output "redis_secret_id" {
  description = "Secret Manager secret ID for REDIS_URL (populate outside Terraform)."
  value       = module.gke.redis_secret_id
}

output "populate_secrets_commands" {
  description = "gcloud commands to add secret versions after terraform apply (never commit keys)."
  value       = module.gke.populate_secrets_commands
}

output "terraform_state_encryption_note" {
  description = "Terraform remote state on GCS is encrypted at rest by default (Google-managed keys). Use a backend encryption_key for CMEK."
  value       = module.gke.terraform_state_encryption_note
}
