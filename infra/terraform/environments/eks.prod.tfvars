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
# Set image_tag to the long git SHA published by CI (type=sha,format=long), e.g.:
# image_tag = "a310c5bf279f3b60a4421d44004af7f4652e6993"
# Do NOT use a digest here — the chart renders repository:tag, not repository@sha256:...
# Secrets: -var="anthropic_api_key=$ANTHROPIC_API_KEY" -var="redis_url=$REDIS_URL"
