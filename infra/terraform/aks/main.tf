# ---------------------------------------------------------------------------
# AKS entry point — AKS managed cluster + Log Analytics + Helm release
#
# Usage:
#   cd infra/terraform/aks
#   terraform init
#   terraform apply -var-file=../environments/azure.dev.tfvars \
#     -var="subscription_id=$ARM_SUBSCRIPTION_ID" \
#     -var="anthropic_api_key=$ANTHROPIC_API_KEY"
#
# Production:
#   terraform apply -var-file=../environments/azure.prod.tfvars \
#     -var="subscription_id=$ARM_SUBSCRIPTION_ID" \
#     -var="anthropic_api_key=$ANTHROPIC_API_KEY" \
#     -var="redis_url=$REDIS_URL"
# ---------------------------------------------------------------------------

module "aks" {
  source = "../modules/aks"

  subscription_id     = var.subscription_id
  resource_group_name = var.resource_group_name
  location            = var.location
  cluster_name        = var.cluster_name
  environment         = var.environment
  kubernetes_version  = var.kubernetes_version
  node_count          = var.node_count
  node_vm_size        = var.node_vm_size
  namespace           = var.namespace
  helm_chart_path     = var.helm_chart_path
  anthropic_api_key   = var.anthropic_api_key
  redis_url           = var.redis_url
  llm_provider        = var.llm_provider

  tags = {
    environment = var.environment
    project     = "langgraph-agent-stack"
    managed-by  = "terraform"
  }
}
