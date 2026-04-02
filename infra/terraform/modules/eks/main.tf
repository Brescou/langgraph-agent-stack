# ---------------------------------------------------------------------------
# EKS module — managed cluster + IRSA + Helm release
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. AWS provider
# ---------------------------------------------------------------------------
provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      environment = var.environment
      managed-by  = "terraform"
      project     = "langgraph-agent-stack"
    }
  }
}

# ---------------------------------------------------------------------------
# 2. Data sources — retrieve default VPC and subnets for simplicity.
#    For production, replace with dedicated VPC/subnet resources.
# ---------------------------------------------------------------------------
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Current AWS account ID — used for ARN construction.
data "aws_caller_identity" "current" {}

# TLS certificate for the EKS OIDC provider — required for IRSA.
data "tls_certificate" "eks" {
  url = aws_eks_cluster.main.identity[0].oidc[0].issuer
}

# ---------------------------------------------------------------------------
# 3. IAM role for the EKS control plane
# ---------------------------------------------------------------------------
resource "aws_iam_role" "eks_cluster" {
  name = "${var.cluster_name}-cluster-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "eks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_cluster_policy" {
  role       = aws_iam_role.eks_cluster.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
}

# ---------------------------------------------------------------------------
# 4. EKS cluster
# ---------------------------------------------------------------------------
resource "aws_eks_cluster" "main" {
  name     = var.cluster_name
  role_arn = aws_iam_role.eks_cluster.arn
  version  = var.eks_version

  vpc_config {
    subnet_ids = data.aws_subnets.default.ids
  }

  # Enable OIDC — required for IRSA.
  enabled_cluster_log_types = ["api", "audit", "authenticator"]

  depends_on = [aws_iam_role_policy_attachment.eks_cluster_policy]
}

# ---------------------------------------------------------------------------
# 5. IAM role for the managed node group
# ---------------------------------------------------------------------------
resource "aws_iam_role" "eks_node" {
  name = "${var.cluster_name}-node-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ec2.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "eks_worker_node" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy"
}

resource "aws_iam_role_policy_attachment" "eks_cni" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy"
}

resource "aws_iam_role_policy_attachment" "eks_ecr_read" {
  role       = aws_iam_role.eks_node.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

# ---------------------------------------------------------------------------
# 6. Managed node group
#    Instance type: t3.medium; scaling: min 1 / max 3 (configurable)
# ---------------------------------------------------------------------------
resource "aws_eks_node_group" "main" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "${var.cluster_name}-nodes"
  node_role_arn   = aws_iam_role.eks_node.arn
  subnet_ids      = data.aws_subnets.default.ids

  instance_types = [var.node_instance_type]

  scaling_config {
    desired_size = var.node_desired_size
    min_size     = var.node_min_size
    max_size     = var.node_max_size
  }

  update_config {
    # Allow one node unavailable during rolling updates.
    max_unavailable = 1
  }

  depends_on = [
    aws_iam_role_policy_attachment.eks_worker_node,
    aws_iam_role_policy_attachment.eks_cni,
    aws_iam_role_policy_attachment.eks_ecr_read,
  ]
}

# ---------------------------------------------------------------------------
# 7. IRSA (IAM Roles for Service Accounts)
#    Binds a Kubernetes service account in the langgraph namespace to an
#    IAM role, enabling pod-level AWS credential scoping without static keys.
# ---------------------------------------------------------------------------

# OIDC identity provider for the cluster.
resource "aws_iam_openid_connect_provider" "eks" {
  url             = aws_eks_cluster.main.identity[0].oidc[0].issuer
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = [data.tls_certificate.eks.certificates[0].sha1_fingerprint]
}

# IAM role assumed by the Kubernetes service account via OIDC federation.
resource "aws_iam_role" "langgraph_irsa" {
  name = "${var.cluster_name}-langgraph-irsa"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRoleWithWebIdentity"
      Principal = {
        Federated = aws_iam_openid_connect_provider.eks.arn
      }
      Condition = {
        StringEquals = {
          "${replace(aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")}:sub" = "system:serviceaccount:${var.namespace}:${var.helm_release_name}-langgraph-agent-stack"
          "${replace(aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

# Minimal policy for the IRSA role — extend as needed (e.g. S3, SSM).
resource "aws_iam_policy" "langgraph_irsa" {
  name        = "${var.cluster_name}-langgraph-irsa-policy"
  description = "Permissions for the langgraph-agent-stack service account."

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        # Allow reading secrets from AWS Secrets Manager if needed.
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret",
        ]
        Resource = "arn:aws:secretsmanager:${var.aws_region}:${data.aws_caller_identity.current.account_id}:secret:langgraph/*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "langgraph_irsa" {
  role       = aws_iam_role.langgraph_irsa.name
  policy_arn = aws_iam_policy.langgraph_irsa.arn
}

# ---------------------------------------------------------------------------
# 8. Kubernetes provider — uses EKS cluster credentials via AWS CLI token
# ---------------------------------------------------------------------------
data "aws_eks_cluster_auth" "main" {
  name = aws_eks_cluster.main.name
}

provider "kubernetes" {
  host                   = aws_eks_cluster.main.endpoint
  token                  = data.aws_eks_cluster_auth.main.token
  cluster_ca_certificate = base64decode(aws_eks_cluster.main.certificate_authority[0].data)
}

# ---------------------------------------------------------------------------
# 9. Helm provider — shares the same EKS credentials
# ---------------------------------------------------------------------------
provider "helm" {
  kubernetes {
    host                   = aws_eks_cluster.main.endpoint
    token                  = data.aws_eks_cluster_auth.main.token
    cluster_ca_certificate = base64decode(aws_eks_cluster.main.certificate_authority[0].data)
  }
}

# ---------------------------------------------------------------------------
# 10. Kubernetes namespace
# ---------------------------------------------------------------------------
resource "kubernetes_namespace" "langgraph" {
  metadata {
    name = var.namespace

    labels = {
      environment = var.environment
      managed-by  = "terraform"
    }

    annotations = {
      # Annotate the namespace with the IRSA role ARN for reference.
      "eks.amazonaws.com/role-arn" = aws_iam_role.langgraph_irsa.arn
    }
  }

  depends_on = [aws_eks_node_group.main]
}

# ---------------------------------------------------------------------------
# 11. Kubernetes secret for the Anthropic API key
# ---------------------------------------------------------------------------
resource "kubernetes_secret" "anthropic_api_key" {
  metadata {
    name      = "langgraph-secrets"
    namespace = kubernetes_namespace.langgraph.metadata[0].name
  }

  type = "Opaque"

  data = {
    ANTHROPIC_API_KEY = var.anthropic_api_key
  }
}

# ---------------------------------------------------------------------------
# 12. Helm release — langgraph-agent-stack
#     Chart version / appVersion from infra/helm/langgraph-agent-stack/Chart.yaml
#     Default image: langgraph-agent-stack:latest (from values.yaml)
# ---------------------------------------------------------------------------
resource "helm_release" "langgraph" {
  name             = "langgraph"
  chart            = var.helm_chart_path
  namespace        = kubernetes_namespace.langgraph.metadata[0].name
  create_namespace = false # Namespace is managed above.

  # Environment-specific values file (values.dev.yaml or values.prod.yaml).
  values = [file("${var.helm_chart_path}/values.${var.environment}.yaml")]

  # Annotate the service account with the IRSA role so pods receive AWS creds.
  set {
    name  = "serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
    value = aws_iam_role.langgraph_irsa.arn
  }

  # LLM provider override.
  set {
    name  = "llm.provider"
    value = var.llm_provider
  }

  # Reference the pre-created secret.
  set {
    name  = "secrets.existingSecret"
    value = kubernetes_secret.anthropic_api_key.metadata[0].name
  }

  depends_on = [kubernetes_namespace.langgraph]
}
