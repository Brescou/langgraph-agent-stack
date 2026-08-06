aws_region         = "us-east-1"
cluster_name       = "langgraph-cluster"
environment        = "prod"
namespace          = "langgraph-agents"
llm_provider       = "anthropic"
eks_version        = "1.31"
node_instance_type = "t3.medium"
node_min_size      = 1
node_max_size      = 3
node_desired_size  = 2
# Set image_tag to a long SHA for reproducible deploys, e.g.:
# image_tag = "sha256:abc123..."
# Secrets: -var="anthropic_api_key=$ANTHROPIC_API_KEY" -var="redis_url=$REDIS_URL"
