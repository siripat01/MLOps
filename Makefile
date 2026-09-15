.PHONY: install test lint format quality-gate dev-stack-up data-pipeline train-pipeline deploy-build-pipeline serve-api clean

install:
	uv sync --extra dev

test:
	uv run pytest --cov=mlops_project --cov-report=term-missing

lint:
	uv run ruff check .

quality-gate: lint test

dev-stack-up:
	docker compose -f infrastructure/docker/docker-compose.yml up -d mysql zenml mlflow minio minio-create-buckets
	uv run python scripts/wait_for_zenml.py

format:
	uv run ruff format .
	uv run ruff check --fix .

data-pipeline: dev-stack-up
	uv run run-data-pipeline

train-pipeline: dev-stack-up
	uv run run-training-pipeline

deploy-build-pipeline:
	uv run run-deployment-pipeline

serve-api:
	docker compose -f infrastructure/docker/docker-compose.yml --profile serving up forecast-api

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist
