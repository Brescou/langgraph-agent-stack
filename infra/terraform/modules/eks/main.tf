# ---------------------------------------------------------------------------
# EKS module — managed cluster + IRSA + Helm release
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. AWS provider
# ---------------------------------------------------------------------------
# NOTE: Provider declarations in modules is a Terraform anti-pattern that
# prevents using count/for_each on the module call.  This is acceptable here
# because each cloud module is used as a standalone root module via its
# entry-point directory (e.g. infra/terraform/eks/).  If you need to compose
# multiple cloud modules in a single root, refactor providers to the root.
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
# 2. Dedicated VPC — private/public subnets, NAT gateway, Internet gateway.
# ---------------------------------------------------------------------------
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name = "${var.cluster_name}-vpc"
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone = data.aws_availability_zones.available.names[count.index]

  tags = {
    Name                                        = "${var.cluster_name}-private-${count.index}"
    "kubernetes.io/role/internal-elb"           = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index + 100)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name                                        = "${var.cluster_name}-public-${count.index}"
    "kubernetes.io/role/elb"                    = "1"
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = "${var.cluster_name}-igw"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${var.cluster_name}-public-rt"
  }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_eip" "nat" {
  domain = "vpc"

  tags = {
    Name = "${var.cluster_name}-nat-eip"
  }
}

resource "aws_nat_gateway" "main" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id

  tags = {
    Name = "${var.cluster_name}-nat"
  }

  depends_on = [aws_internet_gateway.main]
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.main.id
  }

  tags = {
    Name = "${var.cluster_name}-private-rt"
  }
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
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
    subnet_ids              = aws_subnet.private[*].id
    endpoint_private_access = true
    public_access_cidrs     = var.public_access_cidrs
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
  subnet_ids      = aws_subnet.private[*].id

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

# ---------------------------------------------------------------------------
# EBS CSI driver — required for PersistentVolumeClaims (chart persistence)
# ---------------------------------------------------------------------------
resource "aws_iam_role" "ebs_csi" {
  name = "${var.cluster_name}-ebs-csi"

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
          "${replace(aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")}:sub" = "system:serviceaccount:kube-system:ebs-csi-controller-sa"
          "${replace(aws_eks_cluster.main.identity[0].oidc[0].issuer, "https://", "")}:aud" = "sts.amazonaws.com"
        }
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ebs_csi" {
  role       = aws_iam_role.ebs_csi.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
}

resource "aws_eks_addon" "ebs_csi" {
  cluster_name             = aws_eks_cluster.main.name
  addon_name               = "aws-ebs-csi-driver"
  service_account_role_arn = aws_iam_role.ebs_csi.arn

  depends_on = [
    aws_eks_node_group.main,
    aws_iam_role_policy_attachment.ebs_csi,
  ]
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
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
        ]
        Resource = "*"
      },
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
# NOTE: Provider declarations in modules is a Terraform anti-pattern that
# prevents using count/for_each on the module call.  This is acceptable here
# because each cloud module is used as a standalone root module via its
# entry-point directory (e.g. infra/terraform/eks/).  If you need to compose
# multiple cloud modules in a single root, refactor providers to the root.
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
# NOTE: Helm provider 3.x requires nested object syntax (= {}) instead of blocks.
provider "helm" {
  kubernetes = {
    host                   = aws_eks_cluster.main.endpoint
    token                  = data.aws_eks_cluster_auth.main.token
    cluster_ca_certificate = base64decode(aws_eks_cluster.main.certificate_authority[0].data)
  }
}

# ---------------------------------------------------------------------------
# 10. Kubernetes namespace
# ---------------------------------------------------------------------------
resource "kubernetes_namespace_v1" "langgraph" {
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
# 11. Kubernetes secret for the Anthropic API key (and optional Redis URL)
# ---------------------------------------------------------------------------
resource "kubernetes_secret_v1" "langgraph_secrets" {
  metadata {
    name      = "langgraph-secrets"
    namespace = kubernetes_namespace_v1.langgraph.metadata[0].name
  }

  type = "Opaque"

  data = {
    ANTHROPIC_API_KEY = var.anthropic_api_key
    REDIS_URL         = var.redis_url
  }
}

# ---------------------------------------------------------------------------
# 12. Helm release — langgraph-agent-stack
#     Chart version / appVersion from infra/helm/langgraph-agent-stack/Chart.yaml
#     Default image: langgraph-agent-stack:latest (from values.yaml)
# ---------------------------------------------------------------------------
locals {
  values_files = length(var.helm_values_files) > 0 ? var.helm_values_files : concat(
    ["values.cloud.yaml"],
    var.environment == "prod" ? ["values.prod.yaml"] : [],
  )
}

resource "helm_release" "langgraph" {
  name             = "langgraph"
  chart            = var.helm_chart_path
  namespace        = kubernetes_namespace_v1.langgraph.metadata[0].name
  create_namespace = false # Namespace is managed above.

  values = [for f in local.values_files : file("${var.helm_chart_path}/${f}")]

  # Helm provider 3.x: set is now a list of objects.
  set = concat(
    [
      {
        name  = "serviceAccount.annotations.eks\\.amazonaws\\.com/role-arn"
        value = aws_iam_role.langgraph_irsa.arn
      },
      {
        name  = "llm.provider"
        value = var.llm_provider
      },
      {
        name  = "secrets.existingSecret"
        value = kubernetes_secret_v1.langgraph_secrets.metadata[0].name
      },
    ],
    var.image_repository != "" ? [{ name = "image.repository", value = var.image_repository }] : [],
    var.image_tag != "" ? [{ name = "image.tag", value = var.image_tag }] : [],
  )

  depends_on = [
    kubernetes_namespace_v1.langgraph,
    aws_eks_addon.ebs_csi,
  ]
}
