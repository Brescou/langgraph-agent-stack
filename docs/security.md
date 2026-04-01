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
8. [Reporting Vulnerabilities](#8-reporting-vulnerabilities)

---

## 1. What Is Protected by Default

The following controls ship enabled and require no operator action.

### Container security

- The Docker image runs as a non-root user (`appuser`, UID 1001). Processes
  cannot write outside `/app` without explicit volume mounts.
- Multi-stage build: development tools and the `uv` installer are discarded
  before the runtime image is assembled. Only the application code and the
  pre-built virtual environment are copied.
- No secrets are baked into the image. The Anthropic API key and Redis URL are
  injected at runtime via environment variables or Kubernetes Secrets.

### API layer

- **Security HTTP headers** are set on every response:
  - `X-Content-Type-Options: nosniff` — prevents MIME-type sniffing.
  - `X-Frame-Options: DENY` — blocks clickjacking via iframe embedding.
  - `X-XSS-Protection: 1; mode=block` — legacy XSS filter for older browsers.
  - `Referrer-Policy: strict-origin-when-cross-origin` — limits referrer leakage.
  - `Cache-Control: no-store` — prevents caching of LLM responses.
  - The `Server` header is removed to avoid advertising the runtime stack.
- **Rate limiting** (60 requests per minute per client IP, sliding window) is
  enforced on all endpoints except `/health`.
- **Input validation** (`core/security.InputValidator`) rejects queries that
  contain prompt-injection markers, SSRF-style internal endpoint references,
  server-side template injection syntax, path traversal sequences, and null bytes
  before they reach the LLM.
- **API key format validation** at startup: if `ANTHROPIC_API_KEY` does not match
  the `sk-ant-...` pattern a startup warning is emitted and settings loading fails,
  catching misconfigured placeholder values early.

### Logging

- `core/security.sanitize_log_data` masks values whose key names contain `key`,
  `token`, `secret`, `password`, `passwd`, `pwd`, `credential`, or `auth`.
- Agent logs include only the first 120 characters of a query (`query_preview`)
  to prevent PII or malicious payloads from appearing in full in log sinks.

### Dependency management

- `uv sync --frozen --no-dev` in the Dockerfile ensures the exact lockfile is
  used and development dependencies are excluded from the production image.
- The CI pipeline runs `pip-audit` on every push/PR and weekly to catch newly
  disclosed CVEs.

---

## 2. What Requires Configuration

The following items require explicit operator action before deploying to a
non-development environment.

### CORS origins

The default `allow_origins=["*"]` is intentional for a template so it works
out of the box in development. In production, restrict this to your frontend's
origin:

```python
# api/main.py — replace the wildcard
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

The template ships with no authentication layer. All endpoints are publicly
accessible if the service is reachable. For production deployments you should
add one of:

- An API key header check via FastAPI `Depends` + `Security`.
- An OAuth 2.0 / OIDC token validation middleware.
- A Kubernetes `Ingress` with an auth annotation (e.g. oauth2-proxy).

### TLS termination

The application binds on plain HTTP. TLS must be terminated upstream — at the
Kubernetes Ingress controller, a load balancer, or a service mesh (Istio, Linkerd).
Never expose port 8000 directly to the public internet without TLS.

### Rate limit tuning

The default rate limit (60 req/min per IP) may be too permissive or too strict
depending on your traffic profile. Adjust the `RateLimiter` constructor call in
`api/main.py`:

```python
_rate_limiter = RateLimiter(max_requests=20, window_seconds=60.0)
```

For production workloads with multiple replicas, replace the in-memory limiter
with a Redis-backed implementation (e.g. `slowapi` with Redis storage) so limits
are enforced across all pods.

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
| `ANTHROPIC_API_KEY` | Anthropic API key (`sk-ant-...`). Without this the application fails to start. |

### Optional (with defaults)

| Variable | Default | Description |
|---|---|---|
| `MODEL_NAME` | `claude-3-5-sonnet-20241022` | Claude model identifier. |
| `MAX_TOKENS` | `4096` | Maximum tokens per LLM response (1–32768). |
| `MEMORY_BACKEND` | `sqlite` | Persistence backend: `sqlite` or `redis`. |
| `SQLITE_PATH` | `./data/agent_memory.db` | SQLite file path (dev/test only). |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL. Required when `MEMORY_BACKEND=redis`. |
| `API_HOST` | `0.0.0.0` | Bind address. Use `127.0.0.1` for local-only access. |
| `API_PORT` | `8000` | TCP port. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. |
| `ENVIRONMENT` | `development` | Deployment tag surfaced in `/health` and logs. |
| `SEARCH_PROVIDER` | `mock` | Search tool provider: `mock`, `tavily`, or `serpapi`. |
| `TAVILY_API_KEY` | — | Required when `SEARCH_PROVIDER=tavily`. |
| `SERPAPI_API_KEY` | — | Required when `SEARCH_PROVIDER=serpapi`. |

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

The default rate limiter (`core/security.RateLimiter`) is in-memory and
per-process. It is suitable for single-replica deployments and development.

For multi-replica production deployments the in-memory limiter does not share
state across pods. Replace it with a distributed implementation. The
`slowapi` library integrates naturally with FastAPI and supports Redis storage:

```bash
uv add slowapi redis
```

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, storage_uri="redis://redis:6379/0")
app.state.limiter = limiter
```

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

The `.github/workflows/security.yml` pipeline runs three scanners on every push
to `main`, every pull request, and weekly on Monday at 06:00 UTC.

| Job | Tool | What it detects |
|---|---|---|
| `secrets-scan` | gitleaks | Committed credentials, API keys, tokens in git history |
| `dependency-audit` | pip-audit | Known CVEs in Python dependencies (PyPI advisory database) |
| `sast` | bandit | Python SAST: hardcoded passwords, insecure deserialization, subprocess injection, etc. |

All results are uploaded as GitHub Actions artifacts (30-day retention) and, for
`bandit` and `gitleaks`, as SARIF files visible in the GitHub Security tab
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
```

---

## 8. Reporting Vulnerabilities

If you discover a security vulnerability in this template, please report it
responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

### How to report

1. Open a [GitHub Security Advisory](https://github.com/your-org/langgraph-agent-stack/security/advisories/new)
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
