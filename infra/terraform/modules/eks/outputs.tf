# ---------------------------------------------------------------------------
# EKS module outputs
# ---------------------------------------------------------------------------

output "cluster_endpoint" {
  description = "EKS cluster API server endpoint."
  value       = aws_eks_cluster.main.endpoint
}

output "cluster_name" {
  description = "Name of the EKS cluster."
  value       = aws_eks_cluster.main.name
}

output "cluster_certificate_authority" {
  description = "Base64-encoded certificate authority data for the EKS cluster."
  value       = aws_eks_cluster.main.certificate_authority[0].data
  sensitive   = true
}

output "namespace" {
  description = "Kubernetes namespace where the langgraph-agent-stack is deployed."
  value       = kubernetes_namespace.langgraph.metadata[0].name
}

output "helm_release_status" {
  description = "Status of the langgraph Helm release."
  value       = helm_release.langgraph.status
}

output "irsa_role_arn" {
  description = "ARN of the IAM role bound to the langgraph Kubernetes service account via IRSA."
  value       = aws_iam_role.langgraph_irsa.arn
}
