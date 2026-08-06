# REQUIRED before apply:
#   subscription_id   — prefer -var / ARM_SUBSCRIPTION_ID (see header in aks/main.tf)
#   resource_group_name
resource_group_name = "langgraph-rg" # create or reuse
location            = "canadaeast"
cluster_name        = "langgraph-cluster"
environment         = "dev"
kubernetes_version  = "1.29"
node_count          = 2
node_vm_size        = "Standard_D2s_v3"
namespace           = "langgraph-agents"
llm_provider        = "anthropic"
# Secrets via CLI: anthropic_api_key, optional redis_url
