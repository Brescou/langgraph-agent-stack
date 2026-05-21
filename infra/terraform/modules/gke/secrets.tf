# ---------------------------------------------------------------------------
# Secret Manager (containers only) + External Secrets Operator wiring
#
# Prerequisites: External Secrets Operator installed in the cluster.
# Terraform never stores API key values. After apply, populate secrets with:
#   gcloud secrets versions add <secret_id> --data-file=- <<< "$ANTHROPIC_API_KEY"
# ---------------------------------------------------------------------------

locals {
  k8s_service_account_name = "langgraph-workload"
  k8s_secret_name          = "langgraph-secrets"
  anthropic_secret_id      = "${var.environment}-langgraph-anthropic-api-key"
  redis_secret_id          = "${var.environment}-langgraph-redis-url"
}

resource "google_project_service" "secretmanager" {
  provider = google

  project            = var.project_id
  service            = "secretmanager.googleapis.com"
  disable_on_destroy = false
}

resource "google_secret_manager_secret" "anthropic_api_key" {
  provider = google

  project   = var.project_id
  secret_id = local.anthropic_secret_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.secretmanager]
}

resource "google_secret_manager_secret" "redis_url" {
  provider = google

  project   = var.project_id
  secret_id = local.redis_secret_id

  replication {
    auto {}
  }

  depends_on = [google_project_service.secretmanager]
}

resource "google_service_account" "langgraph" {
  provider = google

  account_id   = "langgraph-${var.environment}"
  display_name = "LangGraph agent stack (${var.environment})"
  project      = var.project_id
}

resource "google_project_iam_member" "langgraph_secret_accessor" {
  provider = google

  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.langgraph.email}"
}

resource "kubernetes_service_account_v1" "workload" {
  provider = kubernetes

  metadata {
    name      = local.k8s_service_account_name
    namespace = kubernetes_namespace_v1.langgraph.metadata[0].name
    labels = {
      environment = var.environment
      managed-by  = "terraform"
    }
    annotations = {
      "iam.gke.io/gcp-service-account" = google_service_account.langgraph.email
    }
  }

}

resource "google_service_account_iam_member" "workload_identity" {
  provider = google

  service_account_id = google_service_account.langgraph.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${kubernetes_namespace_v1.langgraph.metadata[0].name}/${local.k8s_service_account_name}]"
}

resource "kubernetes_manifest" "cluster_secret_store" {
  provider = kubernetes

  manifest = {
    apiVersion = "external-secrets.io/v1beta1"
    kind       = "ClusterSecretStore"
    metadata = {
      name = "langgraph-gcp-secret-manager"
    }
    spec = {
      provider = {
        gcpsm = {
          projectID = var.project_id
          auth = {
            workloadIdentity = {
              clusterLocation = var.region
              clusterName       = var.cluster_name
              serviceAccountRef = {
                name      = local.k8s_service_account_name
                namespace = kubernetes_namespace_v1.langgraph.metadata[0].name
              }
            }
          }
        }
      }
    }
  }

  depends_on = [
    kubernetes_service_account_v1.workload,
    google_service_account_iam_member.workload_identity,
  ]
}

resource "kubernetes_manifest" "langgraph_external_secret" {
  provider = kubernetes

  manifest = {
    apiVersion = "external-secrets.io/v1beta1"
    kind       = "ExternalSecret"
    metadata = {
      name      = "langgraph-secrets"
      namespace = kubernetes_namespace_v1.langgraph.metadata[0].name
    }
    spec = {
      refreshInterval = "1h"
      secretStoreRef = {
        name = "langgraph-gcp-secret-manager"
        kind = "ClusterSecretStore"
      }
      target = {
        name           = local.k8s_secret_name
        creationPolicy = "Owner"
      }
      data = [
        {
          secretKey = "ANTHROPIC_API_KEY"
          remoteRef = {
            key = local.anthropic_secret_id
          }
        },
        {
          secretKey = "REDIS_URL"
          remoteRef = {
            key = local.redis_secret_id
          }
        },
      ]
    }
  }

  depends_on = [kubernetes_manifest.cluster_secret_store]
}
