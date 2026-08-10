# ---------------------------------------------------------------------------
# AKS module variables
# ---------------------------------------------------------------------------

variable "subscription_id" {
  description = "Azure Subscription ID (mandatory since AzureRM 4.x)."
  type        = string
}

variable "resource_group_name" {
  description = "Azure Resource Group name."
  type        = string
}

variable "location" {
  description = "Azure region (e.g. canadaeast, eastus)."
  type        = string
  default     = "canadaeast"
}

variable "cluster_name" {
  description = "Name of the AKS cluster."
  type        = string
  default     = "langgraph-cluster"
}

variable "environment" {
  description = "Deployment environment (dev or prod)."
  type        = string
  default     = "dev"
}

variable "kubernetes_version" {
  description = "Kubernetes version for the AKS cluster."
  type        = string
  default     = "1.29"
}

variable "node_count" {
  description = "Initial number of nodes in the default node pool."
  type        = number
  default     = 2
}

variable "node_vm_size" {
  description = "Azure VM size for nodes in the default node pool."
  type        = string
  default     = "Standard_D2s_v3"
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

variable "redis_url" {
  description = "Redis connection URL (optional). Injected as a Kubernetes secret."
  type        = string
  sensitive   = true
  default     = ""
}

variable "llm_provider" {
  description = "LLM provider name (e.g. anthropic, openai, google)."
  type        = string
  default     = "anthropic"
}

variable "tags" {
  description = "Azure tags applied to all resources."
  type        = map(string)
  default     = {}
}

# ---------------------------------------------------------------------------
# API server access
# ---------------------------------------------------------------------------

variable "authorized_ip_ranges" {
  description = "IP CIDRs allowed to reach the AKS API server. Empty list = unrestricted."
  type        = list(string)
  default     = []
}

variable "helm_values_files" {
  description = "Overrides the default Helm overlay list entirely. Empty = derive from environment."
  type        = list(string)
  default     = []
}

variable "image_repository" {
  description = "Optional override for image.repository (empty = use values.cloud.yaml)."
  type        = string
  default     = ""
}

variable "image_tag" {
  description = "Optional override for image.tag (empty = Chart.AppVersion via omitted tag)."
  type        = string
  default     = ""
}
