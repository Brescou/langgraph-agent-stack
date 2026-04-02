# ---------------------------------------------------------------------------
# EKS module variables
# ---------------------------------------------------------------------------

variable "aws_region" {
  description = "AWS region for the EKS cluster."
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "Name of the EKS cluster."
  type        = string
  default     = "langgraph-cluster"
}

variable "environment" {
  description = "Deployment environment (dev or prod)."
  type        = string
  default     = "dev"
}

variable "namespace" {
  description = "Kubernetes namespace for the langgraph-agent-stack."
  type        = string
  default     = "langgraph-agents"
}

variable "helm_chart_path" {
  description = "Path to the langgraph-agent-stack Helm chart directory."
  type        = string
}

variable "anthropic_api_key" {
  description = "Anthropic API key — injected as a Kubernetes secret, never logged."
  type        = string
  sensitive   = true
}

variable "llm_provider" {
  description = "LLM provider name (e.g. anthropic, openai, google)."
  type        = string
  default     = "anthropic"
}

variable "helm_release_name" {
  description = "Name of the Helm release — must match the IRSA service account subject."
  type        = string
  default     = "langgraph"
}

variable "eks_version" {
  description = "Kubernetes version for the EKS cluster."
  type        = string
  default     = "1.31"
}

# ---------------------------------------------------------------------------
# EKS node group sizing
# ---------------------------------------------------------------------------

variable "node_instance_type" {
  description = "EC2 instance type for the managed node group."
  type        = string
  default     = "t3.medium"
}

variable "node_min_size" {
  description = "Minimum number of nodes in the managed node group."
  type        = number
  default     = 1
}

variable "node_max_size" {
  description = "Maximum number of nodes in the managed node group."
  type        = number
  default     = 3
}

variable "node_desired_size" {
  description = "Desired number of nodes in the managed node group."
  type        = number
  default     = 2
}
