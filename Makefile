# ─── Variables ───────────────────────────────────────────────────────────────

PYTHON       := uv run python
UV           := uv
DOCKER_COMPOSE := docker compose -f infra/docker-compose.yml
HELM_CHART   := infra/helm/langgraph-agent-stack

# ─── Phony Targets ───────────────────────────────────────────────────────────

.PHONY: help install run run-ollama \
        test test-cov lint format check \
        docker-build docker-run docker-redis docker-down \
        helm-lint helm-dev helm-prod helm-dry-run helm-uninstall \
        tf-init tf-plan tf-apply \
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

test-cov: ## Run tests with coverage report for agents, core and api
	$(UV) run pytest tests/ --cov=agents --cov=core --cov=api --cov-report=term-missing

lint: ## Check code style with ruff
	$(UV) run ruff check .

format: ## Format source code with black
	$(UV) run black .

check: ## Run linting and formatting checks without modifying files (CI mode)
	$(UV) run ruff check --no-fix .
	$(UV) run black --check .

# ─── Docker ──────────────────────────────────────────────────────────────────

docker-build: ## Build the Docker image tagged as langgraph-agent-stack:latest
	docker build -f infra/Dockerfile -t langgraph-agent-stack:latest .

docker-run: ## Start all services with docker compose (rebuild on change)
	$(DOCKER_COMPOSE) up --build

docker-redis: ## Start all services including the redis profile
	$(DOCKER_COMPOSE) --profile redis up --build

docker-down: ## Stop and remove all docker compose containers
	$(DOCKER_COMPOSE) down

# ─── Helm ────────────────────────────────────────────────────────────────────

helm-lint: ## Lint the Helm chart for errors
	helm lint $(HELM_CHART)

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

tf-init: ## Initialize the Terraform working directory
	terraform -chdir=infra/terraform init

tf-plan: ## Generate and display the Terraform execution plan
	terraform -chdir=infra/terraform plan

tf-apply: ## Apply the Terraform execution plan
	terraform -chdir=infra/terraform apply

# ─── Utilities ───────────────────────────────────────────────────────────────

clean: ## Remove build artifacts, caches and compiled Python files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist"          -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"    -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc"         -delete 2>/dev/null || true
