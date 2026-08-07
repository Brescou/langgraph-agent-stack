# Cloud deploy runbook

End-to-end path from a clean checkout to a `Running` pod on a managed Kubernetes cluster. **EKS** is the primary verification path; GKE and AKS follow the same Helm/image story with extra account-specific tfvars.

For security hardening (Checkov prod gate, Cosign, digest pinning), see [docs/security.md](security.md).

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **AWS CLI** + credentials | EKS path; `aws sts get-caller-identity` should succeed |
| **Terraform** ≥ 1.6 | `terraform version` |
| **kubectl** | Matches cluster version |
| **Helm** ≥ 3 | Used by Terraform modules |
| **Anthropic API key** | Passed at apply: `-var="anthropic_api_key=$ANTHROPIC_API_KEY"` |
| **Budget alert** | Set a cloud billing alert before first apply — dev EKS defaults cost roughly **~$0.25/h** in `us-east-1` (2× `t3.medium` nodes, NAT, EKS control plane). No load balancer is created: the chart uses `ClusterIP` and ingress is off by default |

GKE and AKS need their cloud CLIs (`gcloud`, `az`) and account-specific tfvars filled first (see below).

---

## Container image (GHCR)

The cloud Helm overlay (`infra/helm/langgraph-agent-stack/values.cloud.yaml`) sets:

```yaml
image:
  repository: ghcr.io/brescou/langgraph-agent-stack
  pullPolicy: IfNotPresent
  # tag omitted → Chart AppVersion (currently 0.5.0)
```

So the default cloud deploy pulls **`ghcr.io/brescou/langgraph-agent-stack:0.5.0`** — the chart's `appVersion`, not `:latest`.

### Release tag required (operator step)

The immutable `:0.5.0` image exists only after a maintainer cuts and pushes git tag **`v0.5.0`** (CI publishes semver tags on `v*`). **Immediately after merging the cloud-deploy changes to `main`, cut and push `v0.5.0` before any cloud apply** — the window between merge and tag leaves every cloud deploy in `ImagePullBackOff` because the overlay resolves to `:0.5.0` and that tag is not on GHCR yet.

`latest` is still published on every push to `main`, but the cloud overlay does **not** request it.

### GHCR visibility

**Upstream template (public fork):** make the GHCR package **public** so nodes can pull without credentials:

1. GitHub → your repo → Packages → `langgraph-agent-stack` → Package settings → Change visibility → Public.

**Private fork:** create a pull secret and wire it through Helm:

```bash
kubectl create secret docker-registry ghcr-pull \
  --docker-server=ghcr.io \
  --docker-username=YOUR_GITHUB_USER \
  --docker-password="$GITHUB_TOKEN" \
  --namespace=langgraph-agents

# In tfvars or Helm values:
# image.pullSecrets:
#   - name: ghcr-pull
```

Or pass via Terraform `image_repository` / `image_tag` overrides for a private mirror.

### Production pinning

Pin a long git SHA (not a floating tag) via Terraform:

```bash
terraform apply ... -var="image_tag=abc123def456..."
```

For supply-chain verification (Cosign keyless OIDC), see [§ Supply chain](security.md#8-supply-chain-sbom--image-signing).

---

## EKS (primary path)

### Apply

```bash
cd infra/terraform/eks
terraform init
terraform apply -var-file=../environments/eks.dev.tfvars \
  -var="anthropic_api_key=$ANTHROPIC_API_KEY"
```

`eks.dev.tfvars` supplies region, sizing, and Kubernetes version. Only secrets go on the CLI.

### Verify

```bash
aws eks update-kubeconfig --name langgraph-cluster --region us-east-1
kubectl -n langgraph-agents get pods,pvc
# expect: pod Running, PVC Bound
kubectl -n langgraph-agents get deploy -o wide
# image should show ghcr.io/brescou/langgraph-agent-stack:0.5.0 (or your image_tag override)
```

Optional smoke test (port-forward):

```bash
kubectl -n langgraph-agents port-forward svc/langgraph-agent-stack 8000:8000
curl -s http://localhost:8000/health
```

### Destroy

```bash
cd infra/terraform/eks
terraform destroy -var-file=../environments/eks.dev.tfvars \
  -var="anthropic_api_key=$ANTHROPIC_API_KEY"
```

### Destroy leftover (CloudWatch)

EKS control-plane logging (`enabled_cluster_log_types`) creates a CloudWatch log group **outside Terraform state**. After `terraform destroy`, delete it manually if you want zero residual cost:

```bash
aws logs delete-log-group \
  --log-group-name "/aws/eks/langgraph-cluster/cluster"
```

Adjust the name if you changed `cluster_name` in tfvars.

---

## GKE

Fill **required** fields at the top of `infra/terraform/environments/gke.dev.tfvars` (or `gke.prod.tfvars`):

- `project_id`
- `master_authorized_cidrs` (empty default blocks API access from outside the VPC)

```bash
cd infra/terraform/gke
terraform init
terraform apply -var-file=../environments/gke.dev.tfvars \
  -var="anthropic_api_key=$ANTHROPIC_API_KEY"
```

GKE Autopilot provides disk CSI — no EBS-equivalent addon. Same `values.cloud.yaml` image story as EKS. Install [External Secrets Operator](security.md#3-secret-management) before production `ClusterSecretStore` resources.

---

## AKS

Fill **required** fields in `infra/terraform/environments/azure.dev.tfvars`:

- `subscription_id` (prefer `-var` or `ARM_SUBSCRIPTION_ID`)
- `resource_group_name`

```bash
cd infra/terraform/aks
terraform init
terraform apply -var-file=../environments/azure.dev.tfvars \
  -var="anthropic_api_key=$ANTHROPIC_API_KEY"
```

AKS provides managed disk CSI. Same cloud overlay and image defaults as EKS.

---

## Helm overlays (reference)

| Overlay | Use |
|---------|-----|
| `values.yaml` | Chart defaults |
| `values.dev.yaml` | Local kind/minikube (`pullPolicy: Never`) — **not** used by Terraform cloud path |
| `values.cloud.yaml` | GHCR registry + `IfNotPresent`; tag from AppVersion |
| `values.prod.yaml` | Prod behaviour (replicas, redis, networkPolicy, `pullPolicy: Always`, persistence off) |

Terraform modules layer `values.cloud.yaml` + `values.prod.yaml` when `environment=prod`. Override the list entirely with `helm_values_files` if needed.

---

## Cost note

Dev EKS with default tfvars (`2× t3.medium`, single NAT gateway, public API endpoint) runs roughly **~$0.25/h** in `us-east-1`. There is **no** AWS load balancer: the Service is `ClusterIP` and ingress is disabled. Tear down with `terraform destroy` when finished; remember the CloudWatch log group leftover above.

---

## Related docs

| Doc | Contents |
|-----|----------|
| [docs/security.md](security.md) | Auth, secrets, Checkov prod gate, Cosign verify |
| [README.md](../README.md) | Local quick start, Docker Compose, Helm one-liner |
| [docs/architecture.md](architecture.md) | Platform kernel and pack architecture |
