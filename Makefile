.PHONY: lint format test test-integration typecheck docker-build docker-up db-migrate

lint:
	ruff check .

format:
	ruff format .

test:
	pytest -m "not integration" --tb=short --cov=snoopy --cov-report=term-missing --cov-fail-under=75

test-integration:
	pytest -m integration --tb=short

typecheck:
	mypy snoopy

docker-build:
	docker compose build

docker-up:
	docker compose up -d

db-migrate:
	python -c "from snoopy.db.session import init_db; from snoopy.config import load_config; c = load_config(); init_db(c.storage.database_url); from snoopy.db.migrations import create_all_tables; create_all_tables()"
