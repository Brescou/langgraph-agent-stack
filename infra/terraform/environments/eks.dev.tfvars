aws_region         = "us-east-1"
cluster_name       = "langgraph-cluster"
environment        = "dev"
namespace          = "langgraph-agents"
llm_provider       = "anthropic"
eks_version        = "1.31"
node_instance_type = "t3.medium"
node_min_size      = 1
node_max_size      = 3
node_desired_size  = 2
# Secrets: -var="anthropic_api_key=$ANTHROPIC_API_KEY" (and redis_url if needed)
