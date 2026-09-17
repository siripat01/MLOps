# AutoGluon S3/MinIO + FastAPI Serving Migration Design

## Goal

Remove the legacy model-serving build path and replace it with an explicit split between ZenML/AutoGluon training and a generic FastAPI serving container that downloads a versioned AutoGluon predictor artifact once at startup.

## Current state

- `pipelines/training` trains and evaluates a `TimeSeriesPredictor`, but returns a local/ZenML path and does not publish a portable model artifact URI.
- `pipelines/deployment` re-loads an existing ZenML model, evaluates it again, copies the predictor into a Docker build context, and builds/pushes a model-specific serving image.
- `src/mlops_project/serving` lazily loads a local model through request dependency resolution and has no object-store downloader, checksum verification, or startup lifecycle.
- The root environment combines training, orchestration, and serving dependencies.
- Bento-related application imports are already absent, but repository metadata, compatibility environment variables, and untracked Bento skill material still match the forbidden legacy vocabulary.

## Final architecture

### Training

```text
feature artifact
  -> load_data -> split/preprocess -> AutoGluon train -> evaluate -> quality_gate
  -> package full predictor directory + manifest + sha256
  -> upload `model.tar.gz` to S3/MinIO
  -> return model_uri, model_version, metrics
```

Training does not build an OCI image and does not promote or deploy production.

### Artifact contract

Each uploaded object is a gzip tar archive containing a single `model/` directory and `manifest.json`:

```json
{
  "model_name": "store-sales",
  "model_version": "v19",
  "framework": "autogluon-timeseries",
  "autogluon_version": "1.6.1",
  "python_version": "3.12.x",
  "artifact_sha256": "<sha256 of model.tar.gz>",
  "artifact_uri": "s3://ml-models/store-sales/v19/model.tar.gz"
}
```

The entire AutoGluon predictor directory is packaged; no individual pickle is treated as a complete model.

### Serving

`serving/` is an independent Python project. Its FastAPI lifespan reads `MODEL_URI`, downloads the archive once, verifies the optional checksum, extracts it to a local cache directory, and calls `TimeSeriesPredictor.load(model_path)`. The loaded `AutoGluonModel` is stored in application state. `/ready` returns 503 until this succeeds. Request handling never contacts S3.

### Promotion

Promotion is a small script/workflow, not a training pipeline. It validates the artifact, writes `production.json` containing the selected immutable URI, optionally runs a configured rollout command, waits for `/ready`, and runs a `/predict` smoke test. Rollback writes the previous version's URI back to the pointer.

## Boundaries and error behavior

- `ArtifactDownloader` owns URI parsing, object-store download, local cache, extraction, and checksum verification.
- `AutoGluonModel` owns predictor loading, request-to-`TimeSeriesDataFrame` conversion, prediction, and metadata; it does not know S3 details.
- `serving/app/main.py` owns lifecycle and HTTP status mapping only.
- Invalid URI, missing object, corrupt archive, missing model directory, checksum mismatch, or predictor load failure abort startup and keep `/ready` unavailable.
- `/health` reports process liveness without requiring a loaded model.
- `/predict` returns a validation error for malformed Pydantic input and a server error if inference fails.

## Version compatibility

Training and serving pin the same AutoGluon minor version (`1.6.1`) and record the exact installed version in the manifest. The serving image is model-version independent but code-version dependent.

## Testing and verification

- Unit tests cover URI parsing, checksum verification, archive extraction, cache reuse, predictor loading, readiness before/after startup, health, metadata, valid/invalid prediction payloads, and startup failures.
- Training tests cover complete-directory packaging, manifest contents, checksum, and upload URI.
- Promotion tests cover pointer writes, invalid artifact rejection, rollout/readiness failure, and smoke-test failure.
- Verification includes `pytest`, `ruff`, repository-wide `rg -i "bento|bentoml"`, a clean serving Docker build, and a local container smoke test when Docker is available.

## Migration and cleanup

Keep existing data transformation and AutoGluon training logic. Replace only the artifact publication boundary, serving layer, and deployment release path. Remove model-specific image packaging/build code, legacy serving modules, Bento metadata/skills, and the old tests that assert image-per-model behavior. Do not commit model binaries.
