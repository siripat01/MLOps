.PHONY: install test lint format quality-gate dev-stack-up data-pipeline build-training-image push-training-image train-pipeline promote-model serve-api build-serving-image clean

TRAINING_RUNNER_IMAGE ?= docker.io/siripat007/zenml:training-runner-autogluon-1.6.1-torch2.10

install:
	uv sync --extra dev

test:
	uv run pytest --cov=mlops_project --cov-report=term-missing

lint:
	uv run ruff check .

quality-gate: lint test

dev-stack-up:
	docker compose -f infrastructure/docker/docker-compose.yml up --build
	uv run --env-file=.env.example python scripts/wait_for_zenml.py

format:
	uv run ruff format .
	uv run ruff check --fix .

data-pipeline:
	uv run --env-file=.env run-data-pipeline

build-training-image:
	DOCKER_BUILDKIT=1 docker build \
		-f infrastructure/docker/training-runner.Dockerfile \
		-t $(TRAINING_RUNNER_IMAGE) \
		.

push-training-image: build-training-image
	docker push $(TRAINING_RUNNER_IMAGE)

train-pipeline: push-training-image
	TRAINING_RUNNER_IMAGE=$(TRAINING_RUNNER_IMAGE) uv run --env-file=.env run-training-pipeline

promote-model:
	uv run python scripts/promote_model.py promote --version "$${MODEL_VERSION}" --artifact-uri "$${MODEL_URI}" --pointer-uri "$${PRODUCTION_POINTER_URI}" $${MODEL_SHA256:+--archive-sha256 "$${MODEL_SHA256}"}

deploy-model:
	MODEL_URI=$${MODEL_URI:?MODEL_URI is required} make serve-api

serve-api:
	SERVING_IMAGE=$${SERVING_IMAGE:-autogluon-server:1.0.0} \
	S3_ENDPOINT_URL=$${S3_ENDPOINT_URL:-http://minio:9000} \
	S3_ACCESS_KEY=$${S3_ACCESS_KEY:-minioadmin} \
	S3_SECRET_KEY=$${S3_SECRET_KEY:-local-dev-minio} \
	docker compose -f infrastructure/docker/docker-compose.yml \
		--profile serving up -d --force-recreate forecast-api

build-serving-image:
	docker build -f serving/Dockerfile -t $${SERVING_IMAGE:-autogluon-server:1.0.0} serving

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist
