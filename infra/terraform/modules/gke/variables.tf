# ---------------------------------------------------------------------------
# GKE module variables
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
  default = [{
    cidr_block   = "0.0.0.0/0"
    display_name = "all (restrict in production)"
  }]
}
