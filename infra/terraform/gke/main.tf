# ---------------------------------------------------------------------------
# GKE entry point — GKE Autopilot cluster + Helm release
#
# Usage:
#   cd infra/terraform/gke
#   terraform init
#   terraform apply -var-file=../environments/gke.dev.tfvars
#   # Then populate Secret Manager (see module output populate_secrets_commands)
# ---------------------------------------------------------------------------

module "gke" {
  source = "../modules/gke"

  providers = {
    google     = google
    kubernetes = kubernetes
    helm       = helm
  }

  depends_on = [google_container_cluster.main]

  project_id      = var.project_id
  region          = var.region
  cluster_name    = var.cluster_name
  environment     = var.environment
  namespace       = var.namespace
  helm_chart_path = var.helm_chart_path
  llm_provider    = var.llm_provider
}
