# Architecture

## Training

```text
feature data -> preprocessing -> AutoGluon TimeSeries training
            -> evaluation -> quality gate
            -> complete predictor archive + manifest
            -> S3/MinIO upload -> model URI
```

Training uses the root ZenML environment. It does not build a serving image or update the production pointer.

## Artifact contract

Each model version is stored as `s3://<bucket>/<model>/<version>/model.tar.gz`. The archive contains the complete AutoGluon predictor directory under `model/` and a `manifest.json`. The manifest records the model version, AutoGluon version, Python version, framework, and predictor-directory SHA-256.

## Serving

The independent `serving/` project contains only the FastAPI runtime and model dependencies. FastAPI lifespan downloads and extracts the configured artifact once, verifies the archive and manifest checksums, and calls `TimeSeriesPredictor.load()`. The loaded model remains in application state for all requests.

```text
startup -> ArtifactDownloader -> local cache -> AutoGluonModel.load
request -> Pydantic validation -> AutoGluonModel.predict -> JSON response
```

`/health` is liveness, `/ready` is model readiness, `/metadata` reports the loaded manifest, and `/predict` performs inference.

## Promotion

`scripts/promote_model.py` validates an immutable artifact and updates `production.json`. Rollout tooling consumes that pointer, sets `MODEL_URI` on the unchanged serving image, waits for `/ready`, and smoke-tests `/predict`. Rollback writes an earlier artifact URI; retraining and image rebuilding are not required.

## Compatibility

Training and serving pin AutoGluon Timeseries 1.6.1. The serving image is generic across model versions but must use a compatible AutoGluon version recorded in each manifest.
