.PHONY: install test lint format prepare-data clean

install:
	uv sync --extra dev

test:
	uv run pytest --cov=mlops_project --cov-report=term-missing

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check --fix .

prepare-data:
	uv run prepare-data

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist
