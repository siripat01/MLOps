# AutoGluon S3 FastAPI Serving Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split training from serving, publish complete AutoGluon predictor archives to S3/MinIO, serve them through a generic startup-loaded FastAPI container, and remove the legacy model-specific image build path.

**Architecture:** ZenML remains responsible for training/evaluation/quality gates. A training-only packaging/upload boundary emits an immutable `model.tar.gz` URI and manifest. A separate `serving/` project downloads and loads one artifact during FastAPI lifespan; promotion updates an object-store production pointer and performs rollout checks.

**Tech Stack:** Python 3.12, ZenML 0.96.4, AutoGluon Timeseries 1.6.1, boto3, FastAPI, Uvicorn, Pydantic, pandas, Docker, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-17-autogluon-s3-fastapi-migration-design.md`

## Global Constraints

- Training must end at artifact upload; it must not build a serving image or deploy production.
- Serving must download from S3/MinIO once at startup and never download per request.
- The complete AutoGluon predictor directory must be packaged; `predictor.pkl` alone is insufficient.
- Serving dependencies must not include ZenML, KaggleHub, Optuna, or the old model-build runtime.
- The serving image must be generic across model versions and run as non-root when possible.
- AutoGluon version must be pinned and recorded in `manifest.json`.
- Do not store model binaries in Git.
- Preserve existing data preparation and AutoGluon training behavior unless required by the artifact contract.

### Task 1: Lock the artifact contract with tests

**Files:**
- Create: `tests/unit/test_model_artifact.py`
- Create: `pipelines/training/steps/package_model.py`
- Create: `pipelines/training/steps/upload_model.py`

**Interfaces:**
- `package_model(model_path: Path, model_name: str, model_version: str, artifact_root: Path) -> PackagedModel`
- `PackagedModel.archive_path: Path`, `.manifest_path: Path`, `.artifact_sha256: str`, `.artifact_uri: str | None`
- `upload_model(packaged: PackagedModel, bucket: str, prefix: str, endpoint_url: str | None, access_key: str | None, secret_key: str | None, region: str) -> ModelPublication`

- [ ] Write failing tests for copying the whole predictor directory, manifest fields, tarball checksum, safe extraction layout, and `s3://bucket/name/version/model.tar.gz` URI generation.
- [ ] Run `uv run pytest tests/unit/test_model_artifact.py -q`; confirm failure because the package/upload modules do not exist.
- [ ] Implement deterministic tar creation with `model/` and `manifest.json`, compute SHA-256 over the final archive, and include AutoGluon/Python versions.
- [ ] Implement boto3 `upload_file` with explicit endpoint/credentials and return `ModelPublication(model_uri, model_version, sha256)`.
- [ ] Re-run the focused tests and then `uv run ruff check pipelines/training/steps/package_model.py pipelines/training/steps/upload_model.py tests/unit/test_model_artifact.py`.

### Task 2: Connect packaging/upload to training without changing training logic

**Files:**
- Modify: `pipelines/training/main.py`
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `Makefile`
- Modify: `tests/unit/test_training_pipeline.py`

**Interfaces:**
- Add ZenML steps `package_model` and `upload_model` after `evaluate_model` and `quality_gate`.
- Pipeline output metadata must include `model_uri`, `model_version`, and evaluation metrics.
- Environment names: `MODEL_NAME`, `MODEL_VERSION`, `MODEL_ARTIFACT_PREFIX`, `MODEL_BUCKET`, `S3_ENDPOINT_URL`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_REGION`.

- [ ] Add tests asserting the pipeline graph invokes packaging/upload after quality gate and that no deployment/image step is imported.
- [ ] Run the focused training tests; confirm the new assertions fail.
- [ ] Add `boto3` to the training/root dependency set, wire the new steps, and pass the existing trained predictor directory directly into packaging.
- [ ] Make `MODEL_VERSION` explicit in production runs and use a UTC timestamp fallback for local runs; keep code version independent.
- [ ] Run `uv run pytest tests/unit/test_training_pipeline.py tests/unit/test_model_artifact.py -q`.

### Task 3: Create the independent clean-room serving package

**Files:**
- Create: `serving/pyproject.toml`
- Create: `serving/app/__init__.py`
- Create: `serving/app/config.py`
- Create: `serving/app/schemas.py`
- Create: `serving/app/artifact.py`
- Create: `serving/app/model.py`
- Create: `serving/app/main.py`
- Create: `serving/tests/test_artifact.py`
- Create: `serving/tests/test_api.py`

**Interfaces:**
- `Settings.from_env() -> Settings`
- `ArtifactDownloader.download_and_extract(uri: str, expected_sha256: str | None = None) -> Path`
- `AutoGluonModel.load(model_path: Path, metadata: ModelMetadata) -> AutoGluonModel`
- `AutoGluonModel.predict(request: PredictionRequest) -> PredictionResponse`
- FastAPI app routes: `GET /health`, `GET /ready`, `GET /metadata`, `POST /predict`.

- [ ] Write tests first using a fake S3 client and fake `TimeSeriesPredictor`; cover valid/invalid URI, checksum mismatch, corrupt archive, extraction, cache reuse, readiness before/after lifespan, metadata, valid prediction, and Pydantic rejection.
- [ ] Run `uv run --project serving pytest serving/tests -q`; confirm expected failures.
- [ ] Implement config, downloader, model adapter, schemas, and lifespan state with one load at startup.
- [ ] Ensure `main.py` never imports ZenML or root training modules and `/ready` returns 503 until state has a loaded model.
- [ ] Run serving tests and `uv run --project serving ruff check app tests`.

### Task 4: Dockerize the generic serving image

**Files:**
- Create: `serving/Dockerfile`
- Create: `serving/.dockerignore`
- Create: `serving/README.md`
- Modify: `infrastructure/docker/docker-compose.yml`
- Modify: `Makefile`
- Modify: `.env.example`

**Interfaces:**
- Image command: `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
- Runtime variables: `MODEL_URI`, `MODEL_SHA256`, `MODEL_CACHE_DIR`, `S3_ENDPOINT_URL`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_REGION`, `MODEL_METADATA_URI`.

- [ ] Add a Dockerfile test that asserts no model directory is copied into the image and the image has a non-root `USER` and healthcheck.
- [ ] Implement the generic image with dependency installation before source copy, writable cache directories, non-root user, and `HEALTHCHECK` against `/health`.
- [ ] Update Compose to pass `MODEL_URI` and S3 settings and probe `/ready`; remove image names/fallbacks tied to per-model builds.
- [ ] Run `docker build --no-cache -t autogluon-server:test serving` and inspect the image command/user.

### Task 5: Replace deployment pipeline with promotion/rollback workflow

**Files:**
- Create: `scripts/promote_model.py`
- Create: `tests/unit/test_promotion.py`
- Create: `.github/workflows/promote-model.yml`
- Delete: `pipelines/deployment/main.py`
- Delete: `pipelines/deployment/steps/image.py`
- Delete: `pipelines/deployment/steps/package.py`
- Delete: `pipelines/deployment/steps/model_artifact.py`
- Delete: `pipelines/deployment/steps/model_validation.py`
- Delete: `pipelines/deployment/steps/quality_gate.py`
- Delete: `pipelines/deployment/models.py`

**Interfaces:**
- `promote_model(model_uri: str, version: str, pointer_uri: str, ...) -> None`
- `rollback_model(pointer_uri: str, artifact_uri: str, version: str, ...) -> None`
- Pointer JSON: `{"version": "v19", "artifact_uri": "s3://.../model.tar.gz"}`.

- [ ] Write tests for pointer creation, artifact validation before promotion, rollout command failure, readiness failure, smoke-test failure, and rollback pointer replacement.
- [ ] Run the focused promotion tests and verify they fail before implementation.
- [ ] Implement a small boto3-based command with explicit `promote` and `rollback` subcommands; make rollout and smoke test configurable through environment/CLI arguments.
- [ ] Add a GitHub Actions workflow requiring `model_version`/`model_uri` inputs and never invoking the training pipeline.
- [ ] Run `uv run pytest tests/unit/test_promotion.py -q`.

### Task 6: Remove legacy serving/deployment/Bento material and regenerate locks

**Files:**
- Delete: `src/mlops_project/serving/api.py`
- Delete: `src/mlops_project/serving/forecasting.py`
- Delete: `src/mlops_project/serving/schemas.py`
- Delete: `src/mlops_project/serving/service.py`
- Delete: `tests/unit/test_serving.py`
- Delete: `tests/unit/test_deployment_serving.py`
- Delete: `.agents/skills/bentoml-create-bento/`
- Delete: `.agents/skills/bentoml-containerize/`
- Modify: `skills-lock.json`, README, CI, root dependency files

- [ ] Remove `BENTO_*` aliases and all legacy image/package documentation.
- [ ] Update root scripts so training, serving, and promotion commands point to the new paths.
- [ ] Regenerate `uv.lock` and verify root dependency graph contains no BentoML package.
- [ ] Run `rg -n -i --hidden -g '!**/.git/**' 'bento|bentoml' .`; allow zero matches.

### Task 7: Full verification and handoff

**Files:**
- Modify: `README.md`, `docs/architecture.md`

- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run pytest --cov=mlops_project` and `uv run --project serving pytest serving/tests -q`.
- [ ] Run a clean serving Docker build and local `/health`/`/ready` smoke test with a test artifact when Docker and MinIO are available.
- [ ] Verify the final tree, training flow, serving flow, Docker run example, required environment variables, `/predict` curl, promotion, rollback, and verification commands are documented.
- [ ] Run the final Bento search and inspect `git diff --check` before reporting completion.
