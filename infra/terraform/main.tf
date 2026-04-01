# ---------------------------------------------------------------------------
# Root main.tf — routes to GKE or EKS module based on cloud_provider
# ---------------------------------------------------------------------------

locals {
  use_gke = var.cloud_provider == "gke"
  use_eks = var.cloud_provider == "eks"
}

# ---------------------------------------------------------------------------
# GKE module — activated when cloud_provider = "gke"
# ---------------------------------------------------------------------------
module "gke" {
  source = "./modules/gke"
  count  = local.use_gke ? 1 : 0

  project_id        = var.project_id
  region            = var.region
  cluster_name      = var.cluster_name
  environment       = var.environment
  namespace         = var.namespace
  helm_chart_path   = var.helm_chart_path
  anthropic_api_key = var.anthropic_api_key
  llm_provider      = var.llm_provider
}

# ---------------------------------------------------------------------------
# EKS module — activated when cloud_provider = "eks"
# ---------------------------------------------------------------------------
module "eks" {
  source = "./modules/eks"
  count  = local.use_eks ? 1 : 0

  aws_region        = var.aws_region
  cluster_name      = var.eks_cluster_name
  environment       = var.environment
  namespace         = var.namespace
  helm_chart_path   = var.helm_chart_path
  anthropic_api_key = var.anthropic_api_key
  llm_provider      = var.llm_provider
}
