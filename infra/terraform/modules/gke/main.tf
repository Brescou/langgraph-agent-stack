# ---------------------------------------------------------------------------
# GKE module — Autopilot cluster + Helm release
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. Google provider
# ---------------------------------------------------------------------------
# NOTE: Provider declarations in modules is a Terraform anti-pattern that
# prevents using count/for_each on the module call.  This is acceptable here
# because each cloud module is used as a standalone root module via its
# entry-point directory (e.g. infra/terraform/gke/).  If you need to compose
# multiple cloud modules in a single root, refactor providers to the root.
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

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = var.master_ipv4_cidr_block
  }

  master_authorized_networks_config {
    dynamic "cidr_blocks" {
      for_each = var.master_authorized_cidrs
      content {
        cidr_block   = cidr_blocks.value.cidr_block
        display_name = cidr_blocks.value.display_name
      }
    }
  }

  deletion_protection = var.environment == "prod" ? true : false
}

# ---------------------------------------------------------------------------
# 3. Kubernetes provider — uses GKE cluster credentials
# ---------------------------------------------------------------------------
# NOTE: Provider declarations in modules is a Terraform anti-pattern that
# prevents using count/for_each on the module call.  This is acceptable here
# because each cloud module is used as a standalone root module via its
# entry-point directory (e.g. infra/terraform/gke/).  If you need to compose
# multiple cloud modules in a single root, refactor providers to the root.
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
# 6. Helm release — langgraph-agent-stack
#    Credentials: GCP Secret Manager → ESO ExternalSecret → secrets.existingSecret
#    (see secrets.tf). Never pass API keys via Terraform variables.
# ---------------------------------------------------------------------------
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
      value = local.k8s_secret_name
    },
    {
      name  = "serviceAccount.create"
      value = "false"
    },
    {
      name  = "serviceAccount.name"
      value = local.k8s_service_account_name
    },
  ]

  depends_on = [
    kubernetes_namespace_v1.langgraph,
    kubernetes_service_account_v1.workload,
    kubernetes_manifest.langgraph_external_secret,
  ]
}
