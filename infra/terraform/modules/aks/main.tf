# ---------------------------------------------------------------------------
# AKS module — managed cluster + Log Analytics + Helm release
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 1. Azure provider
# ---------------------------------------------------------------------------
provider "azurerm" {
  features {}
  subscription_id = var.subscription_id
}

# ---------------------------------------------------------------------------
# 2. Resource Group
# ---------------------------------------------------------------------------
resource "azurerm_resource_group" "main" {
  name     = var.resource_group_name
  location = var.location
  tags     = var.tags
}

# ---------------------------------------------------------------------------
# 3. Log Analytics Workspace — Azure Monitor integration
#    Retention: 90 days for prod, 30 days for dev.
# ---------------------------------------------------------------------------
resource "azurerm_log_analytics_workspace" "main" {
  name                = "${var.cluster_name}-logs"
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  sku                 = "PerGB2018"
  retention_in_days   = var.environment == "prod" ? 90 : 30
  tags                = var.tags
}

# ---------------------------------------------------------------------------
# 4. AKS Cluster
#    Identity: System-Assigned Managed Identity (no static credentials).
#    Network: Azure CNI with standard load balancer.
#    Auto-scaling: enabled on default node pool.
#
#    Production note: set prevent_destroy = true in lifecycle block when
#    managing a production cluster to protect against accidental destruction.
# ---------------------------------------------------------------------------
resource "azurerm_kubernetes_cluster" "main" {
  name                = var.cluster_name
  location            = azurerm_resource_group.main.location
  resource_group_name = azurerm_resource_group.main.name
  dns_prefix          = var.cluster_name
  kubernetes_version  = var.kubernetes_version

  default_node_pool {
    name       = "default"
    node_count = var.node_count
    vm_size    = var.node_vm_size

    # OS disk: 50 GB is sufficient for the agent workload.
    os_disk_size_gb = 50

    # VMSS is required for auto-scaling.
    type = "VirtualMachineScaleSets"

    # Auto-scaling bounds differ per environment.
    auto_scaling_enabled = true
    min_count           = var.environment == "prod" ? 2 : 1
    max_count           = var.environment == "prod" ? 10 : 3

    upgrade_settings {
      # Allow 10 % surge capacity during node pool upgrades.
      max_surge = "10%"
    }
  }

  # System-Assigned Managed Identity — no static credentials required.
  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "azure"
    load_balancer_sku = "standard"
    outbound_type     = "loadBalancer"
  }

  # Azure Monitor metrics integration.
  monitor_metrics {}

  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.main.id
  }

  tags = var.tags

  lifecycle {
    # Set prevent_destroy = true for production clusters to prevent accidental
    # destruction via `terraform destroy`.  Cannot be dynamic — change manually.
    prevent_destroy = false
  }
}

# ---------------------------------------------------------------------------
# 5. Kubernetes provider — uses AKS cluster credentials
# ---------------------------------------------------------------------------
provider "kubernetes" {
  host                   = azurerm_kubernetes_cluster.main.kube_config[0].host
  client_certificate     = base64decode(azurerm_kubernetes_cluster.main.kube_config[0].client_certificate)
  client_key             = base64decode(azurerm_kubernetes_cluster.main.kube_config[0].client_key)
  cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.main.kube_config[0].cluster_ca_certificate)
}

# ---------------------------------------------------------------------------
# 6. Helm provider — shares the same AKS credentials
#    NOTE: Helm provider 3.x requires nested object syntax (= {}).
# ---------------------------------------------------------------------------
provider "helm" {
  kubernetes = {
    host                   = azurerm_kubernetes_cluster.main.kube_config[0].host
    client_certificate     = base64decode(azurerm_kubernetes_cluster.main.kube_config[0].client_certificate)
    client_key             = base64decode(azurerm_kubernetes_cluster.main.kube_config[0].client_key)
    cluster_ca_certificate = base64decode(azurerm_kubernetes_cluster.main.kube_config[0].cluster_ca_certificate)
  }
}

# ---------------------------------------------------------------------------
# 7. Kubernetes namespace
# ---------------------------------------------------------------------------
resource "kubernetes_namespace_v1" "langgraph" {
  metadata {
    name = var.namespace

    labels = {
      environment = var.environment
      managed-by  = "terraform"
    }
  }

  depends_on = [azurerm_kubernetes_cluster.main]
}

# ---------------------------------------------------------------------------
# 8. Kubernetes secret for the Anthropic API key (and optional Redis URL)
#    The secret name "langgraph-secrets" matches the Helm chart default:
#    secrets.existingSecret.
# ---------------------------------------------------------------------------
resource "kubernetes_secret_v1" "langgraph_secrets" {
  metadata {
    name      = "langgraph-secrets"
    namespace = kubernetes_namespace_v1.langgraph.metadata[0].name
  }

  # Opaque secrets store arbitrary key-value pairs.
  type = "Opaque"

  data = {
    ANTHROPIC_API_KEY = var.anthropic_api_key
    REDIS_URL         = var.redis_url
  }

  depends_on = [kubernetes_namespace_v1.langgraph]
}

# ---------------------------------------------------------------------------
# 9. Helm release — langgraph-agent-stack
#    Chart version / appVersion from infra/helm/langgraph-agent-stack/Chart.yaml.
#    Default image: langgraph-agent-stack:latest (from values.yaml).
# ---------------------------------------------------------------------------
resource "helm_release" "langgraph" {
  name             = "langgraph"
  chart            = var.helm_chart_path
  namespace        = kubernetes_namespace_v1.langgraph.metadata[0].name
  create_namespace = false # Namespace is managed above.

  # Environment-specific values file (values.dev.yaml or values.prod.yaml).
  values = [file("${var.helm_chart_path}/values.${var.environment}.yaml")]

  # Helm provider 3.x: set is now a list of objects.
  set = [
    {
      name  = "llm.provider"
      value = var.llm_provider
    },
    {
      name  = "secrets.existingSecret"
      value = kubernetes_secret_v1.langgraph_secrets.metadata[0].name
    },
  ]

  depends_on = [
    kubernetes_namespace_v1.langgraph,
    kubernetes_secret_v1.langgraph_secrets,
  ]
}
