# MLOps Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the existing ZenML training and FastAPI serving flow without changing the forecasting contract.

**Architecture:** Keep training, artifact storage, promotion, and serving as separate components. Add validation and metadata at the boundaries, make promotion explicit about pointer update versus rollout, and verify the complete artifact lifecycle with tests.

**Tech Stack:** Python 3.12, ZenML 0.96.4, AutoGluon TimeSeries 1.6.1, FastAPI, Pydantic v2, boto3, Docker Compose, uv, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-17-mlops-hardening-design.md`

## Global Constraints

- Preserve the existing `POST /predict` request shape for valid clients.
- Keep model artifacts immutable and stored under versioned S3 keys.
- Do not embed credentials in images or commit real secrets.
- All production-code behavior changes require a failing test first.
- Run root tests, serving tests, Ruff, Compose config validation, and relevant Docker build checks before completion.

---

### Task 1: Serving request validation

**Files:**
- Modify: `serving/app/schemas.py`
- Modify: `serving/app/model.py`
- Modify: `serving/app/main.py`
- Test: `serving/tests/test_api.py`

- [ ] Write tests for missing horizon rows, mismatched item IDs, duplicate future dates, and a valid request.
- [ ] Run the focused tests and verify the new cases fail.
- [ ] Add model-aware validation and map domain validation errors to HTTP 422.
- [ ] Run the focused tests and then the complete serving suite.

### Task 2: Artifact metadata and cache correctness

**Files:**
- Modify: `pipelines/training/steps/package_model.py`
- Modify: `serving/app/artifact.py`
- Modify: `serving/app/schemas.py`
- Test: `tests/unit/test_model_artifact.py`
- Test: `serving/tests/test_artifact.py`

- [ ] Write tests proving prediction length is in the manifest and a changed URI/checksum invalidates cache.
- [ ] Run tests to verify failure.
- [ ] Add manifest fields and a cache marker tied to URI/checksum.
- [ ] Run artifact tests and verify extraction/checksum behavior.

### Task 3: Quality gate and feature fallback

**Files:**
- Modify: `pipelines/training/main.py`
- Modify: `pipelines/training/steps/load_feature.py`
- Test: `tests/unit/test_training_pipeline.py`
- Test: `tests/unit/test_store_sales_data.py`

- [ ] Add tests for all configured quality thresholds and explicit fallback behavior.
- [ ] Run tests to verify failure.
- [ ] Wire WQL/RMSE thresholds and narrow fallback exceptions.
- [ ] Run root tests and Ruff.

### Task 4: Promotion and deployment boundary

**Files:**
- Modify: `scripts/promote_model.py`
- Modify: `.github/workflows/promote-model.yml`
- Modify: `Makefile`
- Test: `tests/unit/test_promotion.py`

- [ ] Add tests for checksum-aware pointers and explicit promotion/rollback semantics.
- [ ] Run tests to verify failure.
- [ ] Include artifact metadata in the pointer and make rollout an explicit optional command.
- [ ] Run promotion tests and CLI help/config checks.

### Task 5: Docker Compose hardening and documentation

**Files:**
- Modify: `infrastructure/docker/docker-compose.yml`
- Modify: `serving/Dockerfile`
- Modify: `README.md`
- Modify: `.env.example`

- [ ] Add compose/config checks for image defaults, health dependencies, and local-only exposure.
- [ ] Apply safe defaults and pin moving image tags where practical.
- [ ] Update runbook for pipeline, promotion, serving, and smoke tests.
- [ ] Run Compose config validation and serving image build check.

### Task 6: Full verification and model refresh

**Files:**
- No source changes unless verification exposes a defect.

- [ ] Run root and serving tests with coverage.
- [ ] Run Ruff and Docker/Compose checks.
- [ ] Run a fresh training pipeline to create a new versioned artifact.
- [ ] Recreate serving with the new URI and verify `/health`, `/ready`, `/metadata`, and `/predict`.
- [ ] Record the new model URI and verification results.
