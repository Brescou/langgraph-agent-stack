# ---------------------------------------------------------------------------
# Outputs are defined in each cloud-specific entry-point directory:
#
#   infra/terraform/gke/  → module.gke outputs (cluster_endpoint, namespace, ...)
#   infra/terraform/eks/  → module.eks outputs (cluster_endpoint, namespace, irsa_role_arn, ...)
#   infra/terraform/aks/  → module.aks outputs (cluster_endpoint, namespace, managed_identity_principal_id, ...)
#
# This file is intentionally empty — the root directory is not a deployable
# Terraform root module.  See main.tf for usage instructions.
# ---------------------------------------------------------------------------
