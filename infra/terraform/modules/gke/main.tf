# ---------------------------------------------------------------------------
# GKE module — Autopilot cluster + Helm release
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. Google provider
# ---------------------------------------------------------------------------
provider "google" {
  project = var.project_id
  region  = var.region
}

# ---------------------------------------------------------------------------
# 2. GKE Autopilot cluster with Workload Identity
# ---------------------------------------------------------------------------
resource "google_container_cluster" "main" {
  name     = var.cluster_name
  location = var.region

  # Autopilot manages node pools automatically; no manual node pool required.
  enable_autopilot = true

  # Workload Identity allows Kubernetes service accounts to impersonate
  # GCP service accounts without static key files.
  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  deletion_protection = var.environment == "prod" ? true : false
}

# ---------------------------------------------------------------------------
# 3. Kubernetes provider — uses GKE cluster credentials
# ---------------------------------------------------------------------------
provider "kubernetes" {
  host                   = "https://${google_container_cluster.main.endpoint}"
  token                  = data.google_client_config.current.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.main.master_auth[0].cluster_ca_certificate)
}

# Current GCP client credentials (used to authenticate to the cluster).
data "google_client_config" "current" {}

# ---------------------------------------------------------------------------
# 4. Helm provider — shares the same Kubernetes credentials
# ---------------------------------------------------------------------------
# NOTE: Helm provider 3.x requires nested object syntax (= {}) instead of blocks.
provider "helm" {
  kubernetes = {
    host                   = "https://${google_container_cluster.main.endpoint}"
    token                  = data.google_client_config.current.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.main.master_auth[0].cluster_ca_certificate)
  }
}

# ---------------------------------------------------------------------------
# 5. Kubernetes namespace
# ---------------------------------------------------------------------------
resource "kubernetes_namespace_v1" "langgraph" {
  metadata {
    name = var.namespace

    labels = {
      environment = var.environment
      managed-by  = "terraform"
    }
  }

  depends_on = [google_container_cluster.main]
}

# ---------------------------------------------------------------------------
# 6. Kubernetes secret for the Anthropic API key
#    The secret key name matches the Helm chart's expected reference:
#    secrets.anthropicApiKey
# ---------------------------------------------------------------------------
resource "kubernetes_secret_v1" "langgraph_secrets" {
  metadata {
    name      = "langgraph-secrets"
    namespace = kubernetes_namespace_v1.langgraph.metadata[0].name
  }

  type = "Opaque"

  data = {
    ANTHROPIC_API_KEY = var.anthropic_api_key
    REDIS_URL         = var.redis_url
  }
}

# ---------------------------------------------------------------------------
# 7. Helm release — langgraph-agent-stack
#    Chart version and appVersion sourced from Chart.yaml (see infra/helm)
#    Default image: langgraph-agent-stack:latest
#    Default namespace from values.yaml: langgraph-agents
# ---------------------------------------------------------------------------
resource "helm_release" "langgraph" {
  name             = "langgraph"
  chart            = var.helm_chart_path
  namespace        = kubernetes_namespace_v1.langgraph.metadata[0].name
  create_namespace = false # Namespace is managed above.

  # Environment-specific values file (values.dev.yaml or values.prod.yaml).
  values = [file("${var.helm_chart_path}/values.${var.environment}.yaml")]

  # Helm provider 3.x: set is now a list of objects.
  set = [
    {
      name  = "llm.provider"
      value = var.llm_provider
    },
    {
      name  = "secrets.existingSecret"
      value = kubernetes_secret_v1.langgraph_secrets.metadata[0].name
    },
  ]

  depends_on = [kubernetes_namespace_v1.langgraph]
}
