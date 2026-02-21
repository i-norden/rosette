.PHONY: lint format test test-integration typecheck docker-build docker-up db-migrate help clean check-all

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

lint:  ## Run ruff linter
	ruff check .

format:  ## Run ruff formatter
	ruff format .

test:  ## Run unit tests with coverage
	pytest -m "not integration" --tb=short --cov=rosette --cov-report=term-missing --cov-fail-under=75

test-integration:  ## Run integration tests only
	pytest -m integration --tb=short

typecheck:  ## Run mypy type checker
	mypy rosette

format-check:  ## Check code formatting
	ruff format --check .

check-all: lint format-check typecheck test  ## Run all checks

clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .mypy_cache .pytest_cache .ruff_cache __pycache__

docker-build:  ## Build Docker image
	docker compose build

docker-up:  ## Start containers
	docker compose up -d

db-migrate:  ## Run database migrations
	alembic upgrade head
