from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status

from app.artifact import ArtifactDownloader
from app.config import Settings
from app.model import AutoGluonModel
from app.schemas import ModelMetadata, PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings.from_env()
    downloader = ArtifactDownloader(
        settings.cache_dir,
        endpoint_url=settings.s3_endpoint_url,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        region=settings.s3_region,
    )
    model_path, manifest = downloader.download_and_extract(
        settings.model_uri, expected_sha256=settings.model_sha256
    )
    metadata = ModelMetadata(**{**manifest, "artifact_uri": settings.model_uri})
    app.state.model = AutoGluonModel.load(model_path, metadata)
    app.state.metadata = metadata
    yield


app = FastAPI(title="AutoGluon Forecast API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request) -> dict[str, str]:
    if not hasattr(request.app.state, "model"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready"
        )
    return {"status": "ready"}


@app.get("/metadata", response_model=ModelMetadata)
def metadata(request: Request) -> ModelMetadata:
    if not hasattr(request.app.state, "metadata"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready"
        )
    return request.app.state.metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, http_request: Request) -> PredictionResponse:
    if not hasattr(http_request.app.state, "model"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model is not ready"
        )
    try:
        return http_request.app.state.model.predict(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
