# REQUIRED before apply — fill these for your account:
#   project_id
#   master_authorized_cidrs  (empty default = API unreachable from outside VPC)
project_id   = "REPLACE_ME" # e.g. "my-gcp-project"
region       = "us-central1"
cluster_name = "langgraph-cluster"
environment  = "dev"
namespace    = "langgraph-agents"
llm_provider = "anthropic"
master_authorized_cidrs = [{
  cidr_block   = "203.0.113.0/24" # REPLACE with your office/VPN egress
  display_name = "operator"
}]
