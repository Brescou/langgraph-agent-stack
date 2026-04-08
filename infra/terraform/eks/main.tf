# ---------------------------------------------------------------------------
# EKS entry point — EKS managed cluster + IRSA + Helm release
#
# Usage:
#   cd infra/terraform/eks
#   terraform init
#   terraform apply -var-file=../environments/eks.dev.tfvars \
#     -var="anthropic_api_key=$ANTHROPIC_API_KEY"
# ---------------------------------------------------------------------------

module "eks" {
  source = "../modules/eks"

  aws_region         = var.aws_region
  cluster_name       = var.cluster_name
  environment        = var.environment
  namespace          = var.namespace
  helm_chart_path    = var.helm_chart_path
  anthropic_api_key  = var.anthropic_api_key
  redis_url          = var.redis_url
  llm_provider       = var.llm_provider
  helm_release_name  = var.helm_release_name
  eks_version        = var.eks_version
  node_instance_type = var.node_instance_type
  node_min_size      = var.node_min_size
  node_max_size      = var.node_max_size
  node_desired_size  = var.node_desired_size
}
