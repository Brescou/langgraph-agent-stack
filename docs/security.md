# Security Guide — langgraph-agent-stack

This document describes the security model of the template, what is protected by
default, what requires operator configuration, and how to report vulnerabilities.

---

## Table of Contents

1. [What Is Protected by Default](#1-what-is-protected-by-default)
2. [What Requires Configuration](#2-what-requires-configuration)
3. [Secret Management](#3-secret-management)
4. [Required vs Optional Environment Variables](#4-required-vs-optional-environment-variables)
5. [Kubernetes Hardening](#5-kubernetes-hardening)
6. [Rate Limiting and Input Validation](#6-rate-limiting-and-input-validation)
7. [Automated Security Scanning](#7-automated-security-scanning)
   - [Before going to production (Checkov)](#before-going-to-production-checkov)
8. [Supply Chain (SBOM & Image Signing)](#8-supply-chain-sbom--image-signing)
9. [Reporting Vulnerabilities](#9-reporting-vulnerabilities)

---

## 1. What Is Protected by Default

The following controls ship enabled and require no operator action.

### Container security

- The Docker image runs as a non-root user (`appuser`, UID 1001). Processes
  cannot write outside `/app` without explicit volume mounts.
- Multi-stage build: development tools and the `uv` installer are discarded
  before the runtime image is assembled. Only the application code and the
  pre-built virtual environment are copied.
- No secrets are baked into the image. LLM provider API keys and Redis URL are
  injected at runtime via environment variables or Kubernetes Secrets. The
  template supports six LLM providers: `anthropic`, `openai`, `google`,
  `bedrock`, `azure`, and `ollama`.

### API layer

- **Security HTTP headers** are set on every response:
  - `X-Content-Type-Options: nosniff` — prevents MIME-type sniffing.
  - `X-Frame-Options: DENY` — blocks clickjacking via iframe embedding.
  - `Content-Security-Policy: default-src 'self'` — restricts script and resource loading to the same origin (replaces deprecated `X-XSS-Protection`).
  - `Referrer-Policy: strict-origin-when-cross-origin` — limits referrer leakage.
  - `Cache-Control: no-store` — prevents caching of LLM responses.
  - The `Server` header is removed to avoid advertising the runtime stack.
- **Rate limiting** (60 requests per minute per client IP, sliding window) is
  enforced on all endpoints except `/health`.
- **Request body size cap** (`MAX_REQUEST_BODY_BYTES`, default 1 MiB): middleware
  rejects oversized POST/PUT/PATCH bodies with HTTP 413 before JSON parsing
  (checks ``Content-Length`` and stream-bounded reads for chunked uploads).
- **Input validation** (`core/security.InputValidator`) rejects queries that
  contain prompt-injection markers, SSRF-style internal endpoint references,
  server-side template injection syntax, path traversal sequences, and null bytes
  before they reach the LLM.
- **Session ID validation**: `session_id` in `RunRequest` is constrained to a
  maximum of 128 characters and must match `^[a-zA-Z0-9_-]+$`. This prevents
  multi-MB payloads from being persisted to the memory backend.
- **Graceful shutdown**: When the server receives a shutdown signal, all pipeline
  endpoints (`/run`, `/run/stream`, `/research`) immediately return
  `503 Service Unavailable` to prevent new work from starting during drain.
- **Multi-modal content safety**: LLM responses are normalised via
  `_extract_text_content()` before `json.loads()` to prevent `TypeError` when
  models return `list[dict]` content blocks instead of plain strings.
- **API key format validation**: the `validate_api_key_format` utility function
  in `core/security` checks whether an Anthropic key matches the `sk-ant-...`
  pattern. This is a helper available for callers to use — it is not an automatic
  startup check.

### Logging

- `core/security.sanitize_log_data` masks values whose key names contain `key`,
  `token`, `secret`, `password`, `passwd`, `pwd`, `credential`, or `auth`.
- Agent logs include only the first 120 characters of a query (`query_preview`)
  to prevent PII or malicious payloads from appearing in full in log sinks.

### Dependency management

- `uv sync --locked --no-dev` in the Dockerfile ensures the exact lockfile is
  used and development dependencies are excluded from the production image.
- The CI pipeline runs `pip-audit` on every push/PR and weekly to catch newly
  disclosed CVEs.
- On every PR, Syft generates an SPDX SBOM and Trivy scans the built image for
  HIGH/CRITICAL CVEs (`.github/workflows/security.yml`).
- On every push to `main`, the image is published to GHCR with an SPDX SBOM
  attached and a Cosign keyless signature (`.github/workflows/ci.yml` `publish`
  job). See [§ Supply chain](docs/security.md#8-supply-chain-sbom--image-signing).

---

## 2. What Requires Configuration

The following items require explicit operator action before deploying to a
non-development environment.

### CORS origins

The default `allow_origins=["*"]` is intentional for a template so it works
out of the box in development. In production, restrict this to your frontend's
origin:

```python
# api/app.py — replace the wildcard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.example.com"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)
```

Never combine `allow_origins=["*"]` with `allow_credentials=True`.

### Authentication and authorisation

The template includes a **single shared Bearer secret** activated by the `API_KEY`
environment variable. When `API_KEY` is set, every request must include an
`Authorization: Bearer <token>` header whose value matches the configured key.
The comparison uses `hmac.compare_digest` for constant-time evaluation,
preventing timing side-channel attacks.

This is **dev / internal-gateway grade** authentication: one secret for all
callers, no rotation workflow, no scopes, no per-tenant keys, and no structured
audit log keyed by caller identity (only rate-limit bucket separation by token).

The following paths are exempt from authentication so they remain accessible
without a token: `/health`, `/ready`, `/docs`, `/redoc`, `/openapi.json`,
`/metrics`.

Leave `API_KEY` unset to disable authentication entirely (suitable when auth
is enforced upstream). In that mode **all non-exempt routes are open**, including
admin endpoints such as ``GET /sessions/{id}/history`` and
``PATCH /packs/{pack_id}/versions/{version}/weight`` — they declare
``Depends(verify_api_key)`` for OpenAPI clarity but the dependency is a no-op
when ``API_KEY`` is unset.

**Regulated vertical packs** (``talent_screening``, ``contract_reviewer``, etc.)
are registered at startup but **blocked at runtime** unless
``REGULATED_PACKS_ENABLED=true``. Listing packs via ``GET /packs`` still works;
``POST /packs/{pack_id}/run`` returns **403** with an explicit opt-in message.
Enable only after completing each pack's ``COMPLIANCE.md`` programme.

**Multi-tenant production** should not rely on this alone. Add one or more of:

| Approach | When to use |
|----------|-------------|
| OAuth 2.0 / OIDC JWT middleware | User-facing SaaS, fine-grained scopes via claims |
| Named API keys in DB (hash + metadata) | Machine clients, per-tenant keys with revocation |
| Ingress / API gateway auth (oauth2-proxy, etc.) | Centralised policy, minimal app changes |

See also `control_plane/README.md` — tenant quotas and dynamic policy are not
implemented yet.

### TLS termination

The application binds on plain HTTP. TLS must be terminated upstream — at the
Kubernetes Ingress controller, a load balancer, or a service mesh (Istio, Linkerd).
Never expose port 8000 directly to the public internet without TLS.

### Rate limit tuning

The default rate limit (60 req/min per client) may be too permissive or too strict
depending on your traffic profile. Limits are enforced in ``api/middleware.py`` via
``create_rate_limiter()`` from ``core/security.py`` (``RATE_LIMIT_BACKEND=redis`` for multi-replica).

**Behind Kubernetes Ingress / a load balancer**, enable proxy trust so clients are
not all bucketed under the LB IP:

```yaml
# infra/helm/langgraph-agent-stack/values.prod.yaml
config:
  trustProxyHeaders: true
  forwardedAllowIps: "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"  # adjust to your cluster
```

Environment equivalents: ``TRUST_PROXY_HEADERS=true`` and
``FORWARDED_ALLOW_IPS``. The container entrypoint passes the same CIDR list to
``uvicorn --forwarded-allow-ips``.

When ``API_KEY`` is set, authenticated requests are still rate-limited **per
client IP** (the shared secret does not identify individual callers). Anonymous
requests use the same IP-based buckets.

For production workloads with multiple replicas, use ``RATE_LIMIT_BACKEND=redis``
so limits are enforced across all pods.

---

## 3. Secret Management

### Development

Copy `.env.example` to `.env` and populate real values. The `.env` file is
listed in `.gitignore` and must never be committed.

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY to your real key
```

### Staging / Production — Kubernetes

**Do not commit Kubernetes secret manifests with real values.** The Helm chart
manages secrets via `values.yaml` and `existingSecret` references.

The recommended production approach is **External Secrets Operator**, which
pulls secrets from a managed secret store and creates native Kubernetes `Secret`
objects at deploy time.

#### Example: External Secrets Operator with AWS Secrets Manager

1. Install the operator into your cluster:

   ```bash
   helm repo add external-secrets https://charts.external-secrets.io
   helm install external-secrets external-secrets/external-secrets -n external-secrets --create-namespace
   ```

2. Create a `SecretStore` that references your AWS Secrets Manager credentials:

   ```yaml
   apiVersion: external-secrets.io/v1beta1
   kind: SecretStore
   metadata:
     name: aws-secrets
     namespace: langgraph-agents
   spec:
     provider:
       aws:
         service: SecretsManager
         region: eu-west-1
         auth:
           secretRef:
             accessKeyIDSecretRef:
               name: aws-credentials
               key: access-key-id
             secretAccessKeySecretRef:
               name: aws-credentials
               key: secret-access-key
   ```

3. Create an `ExternalSecret` that maps your AWS secret to a Kubernetes `Secret`:

   ```yaml
   apiVersion: external-secrets.io/v1beta1
   kind: ExternalSecret
   metadata:
     name: langgraph-agent-stack-secrets
     namespace: langgraph-agents
   spec:
     refreshInterval: 1h
     secretStoreRef:
       name: aws-secrets
       kind: SecretStore
     target:
       name: langgraph-agent-stack-secrets
       creationPolicy: Owner
     data:
       - secretKey: ANTHROPIC_API_KEY
         remoteRef:
           key: langgraph-agents/production
           property: anthropic_api_key
       - secretKey: REDIS_URL
         remoteRef:
           key: langgraph-agents/production
           property: redis_url
   ```

Alternative secret management solutions:

| Solution | Best for |
|---|---|
| **External Secrets Operator + AWS SM** | AWS-native environments |
| **External Secrets Operator + GCP SM** | GCP-native environments |
| **External Secrets Operator + HashiCorp Vault** | Multi-cloud / on-prem |
| **Sealed Secrets** | GitOps workflows where secrets need to be stored in Git encrypted |
| **SOPS + age/GPG** | Lightweight option; encrypted secret files committed to the repo |

---

## 4. Required vs Optional Environment Variables

### Required

| Variable | Description |
|---|---|
| `ANTHROPIC_API_KEY` | API key for the configured LLM provider. Required when `LLM_PROVIDER=anthropic` (the default). Other providers require their own key (e.g. `OPENAI_API_KEY`, `GOOGLE_API_KEY`). |

### Optional (with defaults)

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_MODEL` | `claude-sonnet-5` | Claude model identifier. |
| `MAX_TOKENS` | `4096` | Maximum tokens per LLM response (1–32768). |
| `MEMORY_BACKEND` | `sqlite` | Persistence backend: `sqlite`, `redis`, or `postgres`. |
| `SQLITE_PATH` | `./data/agent_memory.db` | SQLite file path (dev/test only). |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL. Required when `MEMORY_BACKEND=redis`. |
| `API_HOST` | `0.0.0.0` | Bind address. Use `127.0.0.1` for local-only access. |
| `API_PORT` | `8000` | TCP port. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |
| `ENVIRONMENT` | `development` | Deployment tag surfaced in `/health` and logs. |
| `SEARCH_PROVIDER` | `mock` | Search tool provider: `mock`, `tavily`, or `serpapi`. |
| `TAVILY_API_KEY` | — | Required when `SEARCH_PROVIDER=tavily`. |
| `SERPAPI_API_KEY` | — | Required when `SEARCH_PROVIDER=serpapi`. |
| `THREAD_POOL_MAX_WORKERS` | `4` | Size of the ThreadPoolExecutor for blocking agent calls (1–64). |
| `LLM_REQUEST_TIMEOUT_SECONDS` | `120` | Per-call HTTP timeout for synchronous LLM requests (`llm.invoke`). |
| `MAX_REQUEST_BODY_BYTES` | `1048576` | Max inbound HTTP body size; enforced before JSON parsing. |
| `STREAM_TIMEOUT_SECONDS` | `120` | Wall-clock timeout for SSE streaming runs. |

---

## 5. Kubernetes Hardening

### NetworkPolicy

By default Kubernetes allows unrestricted pod-to-pod traffic. Apply a
`NetworkPolicy` to restrict ingress to the agent pod to only the components
that need it (e.g. the Ingress controller) and to restrict egress to only the
required external endpoints (Anthropic API, Redis).

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: langgraph-agent-stack-netpol
  namespace: langgraph-agents
spec:
  podSelector:
    matchLabels:
      app.kubernetes.io/name: langgraph-agent-stack
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow traffic only from the Ingress controller namespace
    - from:
        - namespaceSelector:
            matchLabels:
              kubernetes.io/metadata.name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000
  egress:
    # Allow DNS resolution
    - ports:
        - protocol: UDP
          port: 53
    # Allow HTTPS to Anthropic API (api.anthropic.com resolves to multiple IPs;
    # use an egress gateway or FQDN policy if your CNI supports it)
    - ports:
        - protocol: TCP
          port: 443
    # Allow Redis (adjust port if using a non-default Redis port)
    - to:
        - podSelector:
            matchLabels:
              app.kubernetes.io/name: redis
      ports:
        - protocol: TCP
          port: 6379
```

### Pod Security

The Kubernetes `Deployment` should enforce:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1001
  runAsGroup: 1001
  seccompProfile:
    type: RuntimeDefault
containers:
  - name: langgraph-agent
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
          - ALL
```

Mount a writable `emptyDir` for the SQLite data directory when using the
SQLite backend with a read-only root filesystem:

```yaml
volumeMounts:
  - name: data
    mountPath: /app/data
volumes:
  - name: data
    emptyDir: {}
```

### Resource Limits

Set CPU and memory limits to prevent a single misbehaving pod from consuming
all cluster resources:

```yaml
resources:
  requests:
    cpu: "250m"
    memory: "256Mi"
  limits:
    cpu: "1000m"
    memory: "1Gi"
```

---

## 6. Rate Limiting and Input Validation

### Rate Limiting

Built-in sliding-window rate limiting lives in ``core/security.py`` and is wired
in ``api/middleware.py`` (limiter created in ``api/lifespan.py``). Backends:

| ``RATE_LIMIT_BACKEND`` | Behaviour |
|---|---|
| ``memory`` (default) | Per-process buckets — fine for single replica / dev |
| ``redis`` | Shared buckets across pods — **required for multi-replica autoscaling** |

Client identity for the limiter:

1. **Default (single shared ``API_KEY``)**: one bucket per client IP. All
   callers share the same Bearer secret, so per-token buckets would collapse
   to a single global limit.
2. **Multi-tenant** (``rate_limit_per_token`` when distinct keys exist): one
   bucket per Bearer token.
3. **Unauthenticated**: one bucket per client IP. With ``TRUST_PROXY_HEADERS=true`` and
   ``FORWARDED_ALLOW_IPS`` matching your Ingress/LB, the IP is taken from
   ``X-Forwarded-For`` (left-most hop) instead of the load-balancer address.

Never set ``TRUST_PROXY_HEADERS`` without a restrictive ``FORWARDED_ALLOW_IPS`` —
otherwise clients can spoof ``X-Forwarded-For``.

### Input Validation

`core/security.InputValidator` is called in both `/run` and `/research` before
the query reaches the LLM. The following patterns are rejected:

| Pattern | Example | Risk |
|---|---|---|
| Prompt injection markers | `ignore all previous instructions` | LLM manipulation |
| Role injection | `</system>` tags | Prompt structure corruption |
| Template injection | `{{ 7*7 }}`, `{% for %}` | SSTI in downstream templating |
| SSRF probes | `http://169.254.169.254/` | Cloud metadata exfiltration |
| Path traversal | `../../etc/passwd` | File system access attempts |
| Null bytes | `\x00` | Parser confusion |

To add custom patterns, extend `_DANGEROUS_PATTERNS` in `core/security.py`:

```python
import re
from core.security import _DANGEROUS_PATTERNS

_DANGEROUS_PATTERNS.append(
    re.compile(r"your-custom-pattern", re.IGNORECASE)
)
```

---

## 7. Automated Security Scanning

The `.github/workflows/security.yml` pipeline runs on every push to `main`, every
pull request, and weekly on Monday at 06:00 UTC.

| Job | Tool | What it detects |
|---|---|---|
| `secrets-scan` | gitleaks | Committed credentials, API keys, tokens in git history |
| `dependency-audit` | pip-audit | Known CVEs in Python dependencies (PyPI advisory database) |
| `sast` | bandit | Python SAST: hardcoded passwords, insecure deserialization, subprocess injection, etc. |
| `image-scan` | Syft + Trivy | SPDX SBOM artifact + HIGH/CRITICAL CVEs in the container image |

All results are uploaded as GitHub Actions artifacts (30-day retention) and, for
`bandit`, `gitleaks`, and `trivy`, as SARIF files visible in the GitHub Security tab
(requires GitHub Advanced Security for private repositories).

To run scans locally:

```bash
# gitleaks
brew install gitleaks   # or: https://github.com/gitleaks/gitleaks/releases
gitleaks detect --source . --verbose

# pip-audit
uv tool install pip-audit
uv run pip-audit

# bandit
uv tool install "bandit[sarif]"
uv tool run bandit --recursive --severity-level medium api/ core/ agents/

# syft (SBOM from a local image)
brew install syft   # or: https://github.com/anchore/syft/releases
syft packages langgraph-agent-stack:local -o spdx-json > sbom.spdx.json

# trivy (container scan)
brew install trivy
trivy image --severity HIGH,CRITICAL langgraph-agent-stack:local
```

### Before going to production (Checkov)

The repository ships **two Checkov profiles**:

| Profile | Config file | Used by | Purpose |
|---|---|---|---|
| **Template / CI** | `.checkov.yaml` | `make infra-check`, CI `infra-lint` job | Passes on stub Terraform modules and chart defaults |
| **Production gate** | `.checkov.prod.yaml` | `make infra-check-prod` | Re-enables cloud hardening checks before real deploys |

The template profile skips **~50 cloud Terraform policies** (AWS EKS, Azure AKS, GCP GKE) with the rationale *“harden in production tfvars”*. That keeps CI green on illustrative modules, but a fork that copies the template and runs `terraform apply` without hardening can silently deploy non-compliant infrastructure (public cluster endpoints, missing audit logging, permissive IAM, etc.).

**The vanilla template is expected to fail `make infra-check-prod`.** Treat a green prod gate as a release criterion for your hardened fork, not for the upstream template itself.

#### Pre-production checklist

1. **Harden Terraform** for your cloud (`infra/terraform/{eks,aks,gke}/` + module tfvars): private API endpoints, control-plane / audit logging, secrets encryption, least-privilege IAM, VPC flow logs, authorized networks, supported Kubernetes versions.
2. **Harden Helm prod overlay** (`infra/helm/langgraph-agent-stack/values.prod.yaml`): pin image **digest** (`@sha256:…`), confirm `pullPolicy: Always`, enable `networkPolicy`, set `secrets.existingSecret`, configure ingress TLS and resource limits.
3. **Install prerequisites** called out in modules (e.g. [External Secrets Operator](#3-secret-management) before applying GKE `ClusterSecretStore` / `ExternalSecret` manifests).
4. **Run the prod gate locally**:

   ```bash
   make infra-check-prod
   ```

5. **Fix every failed check** or document a narrow, reviewed exception in your fork (prefer fixing Terraform/Helm over adding new skips).
6. **Wire the prod gate into your deploy pipeline** (optional): run `CHECKOV_CONFIG=.checkov.prod.yaml make infra-check` on the production branch before `terraform apply` / Helm promote.

#### Kubernetes checks re-enabled in the prod profile

These skips exist in `.checkov.yaml` but **not** in `.checkov.prod.yaml` — your prod overlay must satisfy them:

| Check | Policy (summary) | Prod action |
|---|---|---|
| `CKV_K8S_43` | Container image should use digest, not mutable tag | Pin `image@sha256:…` in CD |
| `CKV_K8S_15` | `imagePullPolicy: Always` | Set in `values.prod.yaml` |
| `CKV2_K8S_6` | NetworkPolicy restricts pod traffic | Enable `networkPolicy` in prod values |

Remaining prod skips (`CKV_K8S_35`, `CKV_K8S_37`, `CKV_K8S_40`) are intentional workload choices (envFrom secrets, non-root UID 1001).

#### Cloud Terraform checks skipped in template CI (re-enabled in prod)

Fix or justify each item in your production Terraform before apply.

**AWS (EKS module — `infra/terraform/modules/eks/`)**

| Check | Policy |
|---|---|
| `CKV2_AWS_11` | VPC flow logging enabled |
| `CKV2_AWS_12` | Default security group restricts all traffic |
| `CKV2_AWS_19` | EIPs allocated to VPC are attached to instances |
| `CKV2_AWS_35` | NAT Gateway used for default route |
| `CKV2_AWS_40` | IAM policies do not grant full IAM privileges |
| `CKV2_AWS_44` | VPC peering routes not overly permissive |
| `CKV2_AWS_56` | Managed `IAMFullAccess` policy not used |
| `CKV_AWS_37` | EKS control plane logging enabled (all types) |
| `CKV_AWS_38` | EKS public endpoint not open to `0.0.0.0/0` |
| `CKV_AWS_39` | EKS public endpoint disabled (prefer private) |
| `CKV_AWS_58` | EKS secrets encryption enabled |
| `CKV_AWS_100` | EKS node group: no SSH from `0.0.0.0/0` |
| `CKV_AWS_130` | Subnets do not auto-assign public IPs |
| `CKV_AWS_339` | EKS runs a supported Kubernetes version |
| `CKV_AWS_41` | No hard-coded AWS keys in provider |
| `CKV_AWS_60`–`CKV_AWS_63` | IAM least privilege (assume role, admin `*`, star actions) |
| `CKV_AWS_274` | No `AdministratorAccess` attachments |
| `CKV_AWS_286`–`CKV_AWS_290` | IAM: no privilege escalation / credential exposure / unconstrained write |
| `CKV_AWS_355` | IAM: no `*` resource for restrictable actions |

**Azure (AKS module — `infra/terraform/modules/aks/`)**

| Checks | Theme |
|---|---|
| `CKV_AZURE_4`–`CKV_AZURE_8` | AKS API access, network profile, RBAC |
| `CKV_AZURE_115`–`CKV_AZURE_117` | Storage account encryption / public access |
| `CKV_AZURE_141`, `CKV_AZURE_143` | Key Vault soft-delete / purge protection |
| `CKV_AZURE_168`–`CKV_AZURE_172` | Diagnostic / audit logging |
| `CKV_AZURE_226`, `CKV_AZURE_227`, `CKV_AZURE_232`, `CKV_AZURE_246` | Network security (NSG, private endpoints) |
| `CKV2_AZURE_29` | Storage accounts restrict public network access |

**GCP (GKE root — `infra/terraform/gke/`)**

| Checks | Theme |
|---|---|
| `CKV_GCP_1`, `CKV_GCP_8` | GCS bucket public access / uniform access |
| `CKV_GCP_7` | Legacy ABAC disabled on GKE |
| `CKV_GCP_12`, `CKV_GCP_13`, `CKV_GCP_18` | GKE private cluster / master authorized networks |
| `CKV_GCP_20`–`CKV_GCP_25` | Node pool hardening (shielded nodes, metadata, scopes) |
| `CKV_GCP_61`, `CKV_GCP_64`–`CKV_GCP_71`, `CKV_GCP_123` | Logging, monitoring, binary authorization, release channel |
| `CKV2_GCP_19` | GKE private nodes / control-plane exposure |

For the exact rule text, run `checkov --list` or see the [Checkov policy index](https://www.checkov.io/5.Policy%20Index/terraform.html).

---

## 8. Supply Chain (SBOM & Image Signing)

On every push to `main`, the CI `publish` job (`.github/workflows/ci.yml`) publishes
a production image to **GitHub Container Registry** and attaches supply-chain
metadata:

| Step | Tool | Output |
|---|---|---|
| Build & push | Docker Buildx | `ghcr.io/<owner>/<repo>:latest` and `:sha` |
| SBOM | [Syft](https://github.com/anchore/syft) via `anchore/sbom-action` | SPDX JSON (artifact + registry attachment) |
| Sign | [Cosign](https://github.com/sigstore/cosign) keyless (GitHub OIDC → Sigstore) | OCI signature on the image digest |

Pull requests do **not** push or sign images. The security workflow still builds
locally and uploads an SPDX SBOM artifact for review.

### Pull the signed image

```bash
# Authenticate to GHCR (read packages scope)
echo "$GITHUB_TOKEN" | docker login ghcr.io -u USERNAME --password-stdin

docker pull ghcr.io/<owner>/langgraph-agent-stack:latest
```

For private repositories, grant the deploying principal `read:packages` on the
repository or organisation.

### Verify the Cosign signature (keyless OIDC)

Install Cosign, then verify against the workflow identity that signed the image:

```bash
IMAGE="ghcr.io/<owner>/langgraph-agent-stack"
DIGEST="$(docker buildx imagetools inspect "${IMAGE}:latest" --format '{{json .}}' | jq -r '.manifest.digest')"

cosign verify \
  --certificate-identity "https://github.com/<owner>/langgraph-agent-stack/.github/workflows/ci.yml@refs/heads/main" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  "${IMAGE}@${DIGEST}"
```

A successful verification confirms the image was built and signed by the CI
workflow on `main` — not that the image is vulnerability-free (use Trivy/SBOM for
that).

### Inspect the attached SBOM

```bash
cosign download sbom "${IMAGE}@${DIGEST}" > sbom.spdx.json
```

Or download the `container-sbom-spdx` artifact from the GitHub Actions run.

### Operator checklist

- Pin deployments to **digest** (`image@sha256:…`) rather than floating `:latest`.
- Re-verify signatures in your deploy pipeline before rolling out.
- Feed SPDX SBOMs into your organisation's dependency/VEX tooling if required
  (e.g. compliance, SBOM inventory).

---

## 9. Reporting Vulnerabilities

If you discover a security vulnerability in this template, please report it
responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

### How to report

1. Open a [GitHub Security Advisory](https://github.com/brescou/langgraph-agent-stack/security/advisories/new)
   in this repository.
2. Include:
   - A description of the vulnerability and its potential impact.
   - Steps to reproduce or a proof-of-concept (if safe to share).
   - The affected files and versions.
   - Any suggested mitigations.

### What to expect

- Acknowledgement within 48 hours.
- An initial assessment and severity rating within 5 business days.
- A patch or mitigation within 30 days for high/critical issues, 90 days for
  medium/low issues.
- Credit in the release notes if you wish to be acknowledged.

### Scope

Security reports are welcomed for:

- Authentication and authorisation bypasses.
- Injection vulnerabilities (prompt injection, command injection, SSRF).
- Sensitive data exposure (secrets in logs, responses, or error messages).
- Dependency vulnerabilities that are not yet captured by `pip-audit`.
- Container or Kubernetes misconfigurations in the provided manifests.

Out of scope:

- Vulnerabilities in third-party services (Anthropic API, Redis, Kubernetes itself).
- Issues in development-only components (mock search tool, SQLite backend) that
  do not affect production deployments.
- Social engineering attacks.
