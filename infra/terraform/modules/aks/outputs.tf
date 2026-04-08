# ---------------------------------------------------------------------------
# AKS module outputs
# ---------------------------------------------------------------------------

output "cluster_name" {
  description = "Name of the AKS cluster."
  value       = azurerm_kubernetes_cluster.main.name
}

output "resource_group_name" {
  description = "Name of the Azure Resource Group."
  value       = azurerm_resource_group.main.name
}

output "cluster_endpoint" {
  description = "AKS API server endpoint."
  value       = azurerm_kubernetes_cluster.main.kube_config[0].host
  sensitive   = true
}

output "kube_config" {
  description = "Raw kubeconfig for kubectl access."
  value       = azurerm_kubernetes_cluster.main.kube_config_raw
  sensitive   = true
}

output "namespace" {
  description = "Kubernetes namespace where the langgraph-agent-stack is deployed."
  value       = kubernetes_namespace_v1.langgraph.metadata[0].name
}

output "helm_release_status" {
  description = "Status of the langgraph Helm release."
  value       = helm_release.langgraph.status
}

output "log_analytics_workspace_id" {
  description = "Resource ID of the Log Analytics workspace."
  value       = azurerm_log_analytics_workspace.main.id
}

output "managed_identity_principal_id" {
  description = "Principal ID of the AKS System-Assigned Managed Identity."
  value       = azurerm_kubernetes_cluster.main.identity[0].principal_id
}
