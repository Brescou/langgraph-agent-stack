# ---------------------------------------------------------------------------
# GKE module outputs
# ---------------------------------------------------------------------------

output "namespace" {
  description = "Kubernetes namespace where the langgraph-agent-stack is deployed."
  value       = kubernetes_namespace_v1.langgraph.metadata[0].name
}

output "helm_release_status" {
  description = "Status of the langgraph Helm release."
  value       = helm_release.langgraph.status
}

output "gcp_service_account_email" {
  description = "GCP service account used by the workload via Workload Identity."
  value       = google_service_account.langgraph.email
}

output "anthropic_secret_id" {
  description = "Secret Manager secret ID for ANTHROPIC_API_KEY (populate outside Terraform)."
  value       = google_secret_manager_secret.anthropic_api_key.secret_id
}

output "redis_secret_id" {
  description = "Secret Manager secret ID for REDIS_URL (populate outside Terraform)."
  value       = google_secret_manager_secret.redis_url.secret_id
}

output "populate_secrets_commands" {
  description = "gcloud commands to add secret versions after terraform apply (never commit keys)."
  value       = <<-EOT
    echo -n "$ANTHROPIC_API_KEY" | gcloud secrets versions add ${local.anthropic_secret_id} --project=${var.project_id} --data-file=-
    echo -n "$REDIS_URL" | gcloud secrets versions add ${local.redis_secret_id} --project=${var.project_id} --data-file=-
  EOT
}

output "terraform_state_encryption_note" {
  description = "Terraform remote state on GCS is encrypted at rest by default (Google-managed keys). Use a backend encryption_key for CMEK."
  value       = "GCS backend encrypts state at rest by default. For customer-managed keys, set encryption_key on the backend block."
}
