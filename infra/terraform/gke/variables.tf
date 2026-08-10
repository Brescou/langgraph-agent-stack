# ---------------------------------------------------------------------------
# GKE entry point variables
# ---------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project ID."
  type        = string
}

variable "region" {
  description = "GCP region for the GKE Autopilot cluster."
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  description = "Name of the GKE Autopilot cluster."
  type        = string
  default     = "langgraph-cluster"
}

variable "environment" {
  description = "Deployment environment (dev or prod)."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "environment must be 'dev' or 'prod'."
  }
}

variable "namespace" {
  description = "Kubernetes namespace for the langgraph-agent-stack."
  type        = string
  default     = "langgraph-agents"
}

variable "helm_chart_path" {
  description = "Path to the langgraph-agent-stack Helm chart directory."
  type        = string
  # Relative to this entry-point directory: infra/terraform/gke/
  default = "../../helm/langgraph-agent-stack"
}

variable "llm_provider" {
  description = "LLM provider name (e.g. anthropic, openai, google)."
  type        = string
  default     = "anthropic"
}

# ---------------------------------------------------------------------------
# Private cluster & network access
# ---------------------------------------------------------------------------

variable "master_ipv4_cidr_block" {
  description = "CIDR block for the GKE master's private IP range."
  type        = string
  default     = "172.16.0.0/28"
}

variable "master_authorized_cidrs" {
  description = "CIDRs allowed to reach the GKE API server."
  type = list(object({
    cidr_block   = string
    display_name = string
  }))
  # No default access: the operator MUST provide their own CIDRs to reach
  # the control plane (e.g. office/VPN egress ranges). An empty list means
  # the GKE API server is unreachable from outside the VPC.
  default = []
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
