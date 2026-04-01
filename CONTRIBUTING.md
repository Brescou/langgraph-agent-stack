# Contributing to langgraph-agent-stack

Thank you for taking the time to contribute. This guide covers everything you need to go from a fresh clone to a merged pull request.

## Prerequisites

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — the project's package manager
- Git

Verify your setup:

```bash
python --version   # Python 3.12.x
uv --version       # uv 0.x.x
```

## Development Setup

**1. Clone the repository**

```bash
git clone https://github.com/brescou/langgraph-agent-stack.git
cd langgraph-agent-stack
```

**2. Install all dependencies (including dev extras)**

```bash
uv sync --all-extras
```

**3. Configure your environment**

```bash
cp .env.example .env
```

Open `.env` and set at minimum:

```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

All other values have working defaults for local development.

**4. Verify the setup**

```bash
uv run pytest --tb=short -q
```

All tests should pass before you start making changes.

## Running the Test Suite

```bash
# Run all tests
uv run pytest

# Run a specific file
uv run pytest tests/test_api.py

# Run with coverage
uv run pytest --cov=. --cov-report=term-missing

# Run only fast unit tests (skip slow integration tests)
uv run pytest -m "not integration"
```

The test suite covers API endpoints, memory backends, security primitives, agent logic, and tools. Tests are fully mocked — no external API calls are made during `pytest`.

## Linting and Formatting

The project uses [ruff](https://docs.astral.sh/ruff/) for linting and import sorting, and [black](https://black.readthedocs.io/) for formatting.

```bash
# Check for lint errors
uv run ruff check .

# Auto-fix lint errors where possible
uv run ruff check . --fix

# Format code
uv run black .

# Check formatting without modifying files (CI mode)
uv run black --check .
```

Both checks run automatically in CI on every push and pull request. Your PR will not be merged if either check fails.

## Code Style

- **Type annotations are required** on all function signatures. Use `from __future__ import annotations` for forward references.
- **No bare `except:`** — always catch a specific exception type or at minimum `Exception`.
- **No module-level singletons** — use `get_settings()` (cached with `@lru_cache`) rather than a module-level `Settings()` instance.
- **Log, don't print** — use the standard `logging` module; never use `print()` in library code.
- **Docstrings** — public classes and functions must have a one-line summary docstring.

Example of preferred style:

```python
from __future__ import annotations

import logging

from core.config import get_settings

logger = logging.getLogger(__name__)


def fetch_result(query: str, max_tokens: int = 512) -> str:
    """Return a summarised result for the given query."""
    settings = get_settings()
    try:
        return _call_llm(query, settings.llm_provider, max_tokens)
    except TimeoutError as exc:
        logger.warning("LLM call timed out: %s", exc)
        raise
```

## Branch Naming Convention

| Type | Pattern | Example |
|------|---------|---------|
| New feature | `feat/<short-description>` | `feat/add-ollama-provider` |
| Bug fix | `fix/<short-description>` | `fix/redis-url-logging` |
| Chore / maintenance | `chore/<short-description>` | `chore/update-ruff-version` |
| Documentation | `docs/<short-description>` | `docs/add-contributing-guide` |
| Tests | `test/<short-description>` | `test/add-graph-unit-tests` |

Branch names must be lowercase with hyphens — no underscores, no uppercase.

## Pull Request Process

1. **Create your branch** from `main`:

   ```bash
   git checkout main && git pull
   git checkout -b feat/your-feature
   ```

2. **Make your changes.** Keep commits focused — one logical change per commit.

3. **Run the full check suite before pushing:**

   ```bash
   uv run ruff check .
   uv run black --check .
   uv run pytest
   ```

4. **Open a pull request** against `main`. Fill in the PR template:
   - What problem does this solve?
   - What approach did you take and why?
   - How did you test it?
   - Are there any follow-up tasks?

5. **Address review feedback** by adding new commits — do not force-push to a PR branch that has been reviewed.

6. **Squash on merge** — the maintainer will squash all PR commits into a single commit when merging.

## Adding a New Agent

1. Create `agents/my_agent.py` inheriting from `BaseAgent` in `agents/base_agent.py`.
2. Implement `run()` and `run_structured()`.
3. Add the agent as a node in `core/graph.py` and wire its edges.
4. Expose it via a new endpoint in `api/main.py`, following the `/research` pattern.
5. Add tests in `tests/test_agents.py`.

## Reporting Issues

- Check existing issues before opening a new one.
- For security vulnerabilities, follow the process in [docs/security.md](docs/security.md) — do not open a public issue.
- For bugs, include: Python version, `uv` version, the full error traceback, and a minimal reproduction.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
