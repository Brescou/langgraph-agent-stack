# ---------------------------------------------------------------------------
# GKE platform — namespace, secrets, Helm (cluster lives in the root module)
# ---------------------------------------------------------------------------
# Providers (google, kubernetes, helm) are passed from the entry-point root
# (infra/terraform/gke/) so this module can be used with for_each.

# ---------------------------------------------------------------------------
# Kubernetes namespace
# ---------------------------------------------------------------------------
resource "kubernetes_namespace_v1" "langgraph" {
  provider = kubernetes

  metadata {
    name = var.namespace

    labels = {
      environment = var.environment
      managed-by  = "terraform"
    }
  }
}

# ---------------------------------------------------------------------------
# Helm release — langgraph-agent-stack
# ---------------------------------------------------------------------------
resource "helm_release" "langgraph" {
  provider = helm

  name             = "langgraph"
  chart            = var.helm_chart_path
  namespace        = kubernetes_namespace_v1.langgraph.metadata[0].name
  create_namespace = false

  values = [file("${var.helm_chart_path}/values.${var.environment}.yaml")]

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
