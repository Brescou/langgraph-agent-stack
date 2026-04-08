# ---------------------------------------------------------------------------
# This directory contains reusable modules only.
# Each cloud provider has its own entry-point directory:
#
#   infra/terraform/gke/   — Google Kubernetes Engine (Autopilot)
#   infra/terraform/eks/   — Amazon Elastic Kubernetes Service
#   infra/terraform/aks/   — Azure Kubernetes Service
#
# To deploy, cd into the appropriate directory and run terraform there:
#
#   # GKE
#   cd gke && terraform init && terraform apply \
#     -var-file=../environments/gke.dev.tfvars \
#     -var="anthropic_api_key=$ANTHROPIC_API_KEY"
#
#   # EKS
#   cd eks && terraform init && terraform apply \
#     -var-file=../environments/eks.dev.tfvars \
#     -var="anthropic_api_key=$ANTHROPIC_API_KEY"
#
#   # AKS
#   cd aks && terraform init && terraform apply \
#     -var-file=../environments/azure.dev.tfvars \
#     -var="subscription_id=$ARM_SUBSCRIPTION_ID" \
#     -var="anthropic_api_key=$ANTHROPIC_API_KEY"
#
# See each directory's main.tf for the full usage instructions.
# ---------------------------------------------------------------------------
