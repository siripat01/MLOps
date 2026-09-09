.PHONY: install test lint format quality-gate data-pipeline train-pipeline deploy-build-pipeline serve-api clean

install:
	uv sync --extra dev

test:
	uv run pytest --cov=mlops_project --cov-report=term-missing

lint:
	uv run ruff check .

quality-gate: lint test

format:
	uv run ruff format .
	uv run ruff check --fix .

data-pipeline:
	uv run run-data-pipeline

train-pipeline:
	uv run run-training-pipeline

deploy-build-pipeline:
	uv run run-deployment-pipeline

serve-api:
	docker compose -f infrastructure/docker/docker-compose.yml up forecast-api

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist
