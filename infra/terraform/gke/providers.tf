# Providers are declared at the entry-point root so the GKE module can be
# composed with for_each alongside other cloud modules.

provider "google" {
  project = var.project_id
  region  = var.region
}

data "google_client_config" "current" {}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.main.endpoint}"
  token                  = data.google_client_config.current.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.main.master_auth[0].cluster_ca_certificate)
}

provider "helm" {
  kubernetes = {
    host                   = "https://${google_container_cluster.main.endpoint}"
    token                  = data.google_client_config.current.access_token
    cluster_ca_certificate = base64decode(google_container_cluster.main.master_auth[0].cluster_ca_certificate)
  }
}
