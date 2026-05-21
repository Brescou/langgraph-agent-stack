# GKE Autopilot cluster (root-owned so kubernetes/helm providers can be configured here).

resource "google_container_cluster" "main" {
  name     = var.cluster_name
  location = var.region

  enable_autopilot = true

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = var.master_ipv4_cidr_block
  }

  master_authorized_networks_config {
    dynamic "cidr_blocks" {
      for_each = var.master_authorized_cidrs
      content {
        cidr_block   = cidr_blocks.value.cidr_block
        display_name = cidr_blocks.value.display_name
      }
    }
  }

  deletion_protection = var.environment == "prod" ? true : false

  lifecycle {
    prevent_destroy = var.environment == "prod"
  }
}
