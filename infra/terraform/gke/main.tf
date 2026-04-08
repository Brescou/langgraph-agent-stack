# ---------------------------------------------------------------------------
# GKE entry point — GKE Autopilot cluster + Helm release
#
# Usage:
#   cd infra/terraform/gke
#   terraform init
#   terraform apply -var-file=../environments/gke.dev.tfvars \
#     -var="anthropic_api_key=$ANTHROPIC_API_KEY"
# ---------------------------------------------------------------------------

module "gke" {
  source = "../modules/gke"

  project_id        = var.project_id
  region            = var.region
  cluster_name      = var.cluster_name
  environment       = var.environment
  namespace         = var.namespace
  helm_chart_path   = var.helm_chart_path
  anthropic_api_key = var.anthropic_api_key
  redis_url         = var.redis_url
  llm_provider      = var.llm_provider
}
