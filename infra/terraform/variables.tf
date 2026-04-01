# ---------------------------------------------------------------------------
# Global variables — shared across GKE and EKS modules
# ---------------------------------------------------------------------------

variable "cloud_provider" {
  description = "Target cloud provider for the cluster. Accepted values: 'gke' or 'eks'."
  type        = string
  default     = "gke"

  validation {
    condition     = contains(["gke", "eks"], var.cloud_provider)
    error_message = "cloud_provider must be 'gke' or 'eks'."
  }
}

variable "environment" {
  description = "Deployment environment. Accepted values: 'dev' or 'prod'."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "prod"], var.environment)
    error_message = "environment must be 'dev' or 'prod'."
  }
}

# ---------------------------------------------------------------------------
# LLM / application secrets — never hardcoded
# ---------------------------------------------------------------------------

variable "anthropic_api_key" {
  description = "Anthropic API key injected into the Helm release as a Kubernetes secret. Must never be stored in plaintext."
  type      = string
  sensitive = true
}

variable "llm_provider" {
  description = "LLM provider used by the agent stack (e.g. anthropic, openai, google)."
  type        = string
  default     = "anthropic"
}

# ---------------------------------------------------------------------------
# Kubernetes / Helm
# ---------------------------------------------------------------------------

variable "namespace" {
  description = "Kubernetes namespace where the langgraph-agent-stack is deployed."
  type        = string
  default     = "langgraph-agents"
}

variable "helm_chart_path" {
  description = "Relative or absolute path to the langgraph-agent-stack Helm chart directory."
  type        = string
  default     = "../helm/langgraph-agent-stack"
}

# ---------------------------------------------------------------------------
# GKE-specific variables
# ---------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project ID. Required when cloud_provider = 'gke'."
  type        = string
  default     = ""
}

variable "region" {
  description = "GCP region for the GKE Autopilot cluster."
  type        = string
  default     = "us-central1"
}

variable "cluster_name" {
  description = "Name of the GKE cluster."
  type        = string
  default     = "langgraph-cluster"
}

# ---------------------------------------------------------------------------
# EKS-specific variables
# ---------------------------------------------------------------------------

variable "aws_region" {
  description = "AWS region for the EKS cluster."
  type        = string
  default     = "us-east-1"
}

variable "eks_cluster_name" {
  description = "Name of the EKS cluster."
  type        = string
  default     = "langgraph-cluster"
}
