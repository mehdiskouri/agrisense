# ============================================================================
# AgriSense — Makefile
# ============================================================================
.DEFAULT_GOAL := help
.PHONY: help dev install lint format typecheck test test-julia seed migrate docker-up docker-down clean

PYTHON ?= python3
JULIA  ?= julia

# ── Help ───────────────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ── Development ────────────────────────────────────────────────────────────
install: ## Install Python + Julia dependencies
	$(PYTHON) -m pip install -e ".[dev]"
	$(JULIA) --project=core/AgriSenseCore -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

dev: ## Start uvicorn with hot reload
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# ── Quality ────────────────────────────────────────────────────────────────
lint: ## Run ruff linter
	ruff check app/ tests/ scripts/

format: ## Auto-format with ruff
	ruff format app/ tests/ scripts/
	ruff check --fix app/ tests/ scripts/

typecheck: ## Run mypy type checker
	mypy app/ --ignore-missing-imports

# ── Testing ────────────────────────────────────────────────────────────────
test: ## Run Python test suite
	pytest tests/ -x --tb=short -v

test-julia: ## Run Julia test suite
	$(JULIA) --project=core/AgriSenseCore -e 'using Pkg; Pkg.test()'

test-all: test test-julia ## Run all tests

# ── Database ───────────────────────────────────────────────────────────────
migrate: ## Run Alembic migrations
	alembic upgrade head

migrate-new: ## Create a new migration (usage: make migrate-new MSG="description")
	alembic revision --autogenerate -m "$(MSG)"

seed: ## Seed database with synthetic data
	$(PYTHON) scripts/seed_db.py

# ── Docker ─────────────────────────────────────────────────────────────────
docker-up: ## Start all services via docker-compose
	docker compose up --build -d

docker-down: ## Stop all services
	docker compose down

docker-logs: ## Tail API logs
	docker compose logs -f api

# ── Cleanup ────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/ htmlcov/ .coverage
