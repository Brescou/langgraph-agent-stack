# ---------------------------------------------------------------------------
# Root outputs — resolves to whichever module is active
# ---------------------------------------------------------------------------

output "cluster_endpoint" {
  description = "Kubernetes API server endpoint for the active cluster (GKE or EKS)."
  value = (
    var.cloud_provider == "gke"
    ? module.gke[0].cluster_endpoint
    : module.eks[0].cluster_endpoint
  )
  sensitive = false
}

output "namespace" {
  description = "Kubernetes namespace where the langgraph-agent-stack is deployed."
  value = (
    var.cloud_provider == "gke"
    ? module.gke[0].namespace
    : module.eks[0].namespace
  )
}

output "helm_release_status" {
  description = "Status of the langgraph Helm release in the active cluster."
  value = (
    var.cloud_provider == "gke"
    ? module.gke[0].helm_release_status
    : module.eks[0].helm_release_status
  )
}
