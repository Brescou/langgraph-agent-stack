terraform {
  required_version = ">= 1.6"

  required_providers {
    google = {
      source                = "hashicorp/google"
      version               = "~> 7.0"
      configuration_aliases = [google]
    }
    helm = {
      source                = "hashicorp/helm"
      version               = "~> 3.1"
      configuration_aliases = [helm]
    }
    kubernetes = {
      source                = "hashicorp/kubernetes"
      version               = "~> 3.0"
      configuration_aliases = [kubernetes]
    }
  }
}
