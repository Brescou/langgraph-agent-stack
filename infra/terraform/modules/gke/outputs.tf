# ---------------------------------------------------------------------------
# GKE module outputs
# ---------------------------------------------------------------------------

output "cluster_endpoint" {
  description = "GKE cluster API server endpoint."
  value       = google_container_cluster.main.endpoint
}

output "cluster_name" {
  description = "Name of the GKE Autopilot cluster."
  value       = google_container_cluster.main.name
}

output "namespace" {
  description = "Kubernetes namespace where the langgraph-agent-stack is deployed."
  value       = kubernetes_namespace.langgraph.metadata[0].name
}

output "helm_release_status" {
  description = "Status of the langgraph Helm release."
  value       = helm_release.langgraph.status
}
