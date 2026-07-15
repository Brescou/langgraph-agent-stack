# ─── Variables ───────────────────────────────────────────────────────────────

PYTHON       := uv run python
UV           := uv
DOCKER_COMPOSE := docker compose -f infra/docker-compose.yml
HELM_CHART   := infra/helm/langgraph-agent-stack

# ─── Phony Targets ───────────────────────────────────────────────────────────

.PHONY: help install run run-ollama \
        test test-cov eval eval-ci lint format typecheck check check-security \
        docker-build docker-run docker-redis docker-down docker-smoke \
        helm-lint helm-dev helm-prod helm-dry-run helm-uninstall infra-check \
        tf-init tf-plan tf-apply tf-fmt \
        clean

.DEFAULT_GOAL := help

# ─── Help ────────────────────────────────────────────────────────────────────

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

# ─── Development ─────────────────────────────────────────────────────────────

install: ## Install all dependencies including optional extras
	$(UV) sync --all-extras

run: ## Start the API server with hot reload on port 8000
	$(UV) run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

run-ollama: ## Start the API server using Ollama as LLM provider
	LLM_PROVIDER=ollama $(UV) run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# ─── Quality ─────────────────────────────────────────────────────────────────

test: ## Run the test suite with verbose output
	$(UV) run pytest tests/ -v

eval: ## Run the golden-dataset pack evaluations (deterministic, no network)
	$(UV) run python -m evals --all

eval-ci: ## Same gate as CI (JSON + thresholds); LLM_PROVIDER=mock
	LLM_PROVIDER=mock SEARCH_PROVIDER=mock $(UV) run python -m evals --all --json --thresholds

test-cov: ## Run tests with coverage report (kernel, API, domain packs, connectors)
	$(UV) run pytest tests/ \
		--cov=agents --cov=core --cov=api \
		--cov=pack_kernel --cov=domain_packs --cov=connectors --cov=control_plane \
		--cov-report=term-missing

lint: ## Check code style with ruff
	$(UV) run ruff check .

format: ## Format source code with ruff
	$(UV) run ruff format .

typecheck: ## Run pyright (CI typecheck job)
	$(UV) run pyright

check: ## Lint, format check, and typecheck (CI lint + typecheck jobs)
	$(UV) run ruff check .
	$(UV) run ruff format --check .
	$(UV) run pyright

check-security: ## Bandit + pip-audit gates (CI security.yml, no container scan)
	$(UV) run bandit --recursive --format screen --severity-level high --confidence-level high --exclude .venv,tests api/ core/ agents/ pack_kernel/ domain_packs/ connectors/ control_plane/
	$(UV) export --frozen --no-dev --extra anthropic --no-hashes -o .pip-audit-requirements.txt
	$(UV) run pip-audit -r .pip-audit-requirements.txt --progress-spinner off
	@rm -f .pip-audit-requirements.txt

# ─── Docker ──────────────────────────────────────────────────────────────────

docker-build: ## Build the Docker image tagged as langgraph-agent-stack:latest
	docker build -f infra/Dockerfile -t langgraph-agent-stack:latest .

docker-run: ## Start all services with docker compose (rebuild on change)
	$(DOCKER_COMPOSE) up --build

docker-redis: ## Start all services including the redis profile
	$(DOCKER_COMPOSE) --profile redis up --build

docker-down: ## Stop and remove all docker compose containers
	$(DOCKER_COMPOSE) down

docker-smoke: ## Run Docker smoke test (build + /health + /docs + non-root)
	bash tests/smoke_test_docker.sh

# ─── Helm ────────────────────────────────────────────────────────────────────

helm-lint: ## Lint the Helm chart for errors
	helm lint $(HELM_CHART)

infra-check: ## Helm/Terraform DevSecOps (checkov template profile, kubeconform, kube-linter)
	CHECKOV_CMD="uv tool run checkov" bash scripts/infra-devsecops.sh

infra-check-prod: ## Stricter Checkov prod gate (.checkov.prod.yaml — expect failures on vanilla template)
	CHECKOV_CMD="uv tool run checkov" CHECKOV_CONFIG="$(CURDIR)/.checkov.prod.yaml" bash scripts/infra-devsecops.sh

helm-dev: ## Deploy to the dev environment via Helm
	helm upgrade --install langgraph $(HELM_CHART) \
		-f $(HELM_CHART)/values.dev.yaml \
		--namespace langgraph-agents \
		--create-namespace

helm-prod: ## Deploy to the production environment via Helm
	helm upgrade --install langgraph $(HELM_CHART) \
		-f $(HELM_CHART)/values.prod.yaml \
		--namespace langgraph-agents \
		--create-namespace

helm-dry-run: ## Simulate a Helm install without applying changes
	helm install --dry-run --generate-name $(HELM_CHART)

helm-uninstall: ## Uninstall the Helm release from the langgraph-agents namespace
	helm uninstall langgraph -n langgraph-agents

# ─── Terraform ───────────────────────────────────────────────────────────────
# Each cloud has its own entry point: infra/terraform/{gke,eks,aks}/
# Set TF_CLOUD to the target cloud (default: gke).

TF_CLOUD ?= gke

tf-init: ## Initialize the Terraform working directory
	terraform -chdir=infra/terraform/$(TF_CLOUD) init

tf-plan: ## Generate and display the Terraform execution plan
	terraform -chdir=infra/terraform/$(TF_CLOUD) plan

tf-apply: ## Apply the Terraform execution plan
	terraform -chdir=infra/terraform/$(TF_CLOUD) apply

tf-fmt: ## Check Terraform formatting (all modules)
	terraform -chdir=infra/terraform fmt -check -recursive

# ─── Utilities ───────────────────────────────────────────────────────────────

clean: ## Remove build artifacts, caches and compiled Python files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist"          -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"    -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"         -delete 2>/dev/null || true
