from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, status

from app.artifact import ArtifactDownloader
from app.config import Settings
from app.feedback import FeedbackStore, compute_feedback_metrics
from app.model import AutoGluonModel
from app.monitoring import MonitoringRecorder, WebhookAlerter, request_observations
from app.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    ModelMetadata,
    PredictionRequest,
    PredictionResponse,
)

monitoring = MonitoringRecorder()
MAX_PREDICTION_CACHE_SIZE = 10_000


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings.from_env()
    app.state.monitoring = monitoring
    monitoring.alerter = WebhookAlerter(settings.monitoring_alert_webhook)
    monitoring.drift_threshold = settings.monitoring_drift_threshold
    monitoring.latency_threshold_seconds = settings.monitoring_latency_threshold
    app.state.monitoring.set_model_ready(False)
    app.state.feedback_store = FeedbackStore(settings.monitoring_feedback_path)
    app.state.feedback_store_path = settings.monitoring_feedback_path
    app.state.predictions = {}
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
    app.state.monitoring.set_model_ready(True)
    yield


app = FastAPI(title="AutoGluon Forecast API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    request.state.request_id = request_id
    started = perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        metadata = getattr(request.app.state, "metadata", None)
        model_version = getattr(metadata, "model_version", "unknown")
        getattr(request.app.state, "monitoring", monitoring).record_request(
            route=request.url.path,
            status=500,
            model_version=model_version,
            duration_seconds=perf_counter() - started,
        )
        raise

    metadata = getattr(request.app.state, "metadata", None)
    model_version = getattr(metadata, "model_version", "unknown")
    recorder = getattr(request.app.state, "monitoring", monitoring)
    recorder.record_request(
        route=request.url.path,
        status=response.status_code,
        model_version=model_version,
        duration_seconds=perf_counter() - started,
    )
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def metrics(request: Request) -> Response:
    recorder = getattr(request.app.state, "monitoring", monitoring)
    payload, content_type = recorder.render()
    return Response(content=payload, media_type=content_type.split(";", 1)[0])


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
        result = http_request.app.state.model.predict(request)
        if isinstance(result, dict):
            result = PredictionResponse.model_validate(result)
        request_id = http_request.state.request_id
        metadata = getattr(http_request.app.state, "metadata", None)
        model_version = getattr(metadata, "model_version", "unknown")
        recorder = getattr(http_request.app.state, "monitoring", monitoring)
        recorder.record_prediction(
            model_version=model_version,
            observations=request_observations(request),
            prediction_count=len(result.predictions),
            baseline=getattr(metadata, "monitoring_baseline", {}),
        )
        predictions = getattr(http_request.app.state, "predictions", {})
        http_request.app.state.predictions = predictions
        if len(predictions) >= MAX_PREDICTION_CACHE_SIZE:
            predictions.pop(next(iter(predictions)))
        predictions[request_id] = {
            "model_version": model_version,
            "predictions": {
                (point.item_id, point.timestamp.isoformat()): point.values.get(
                    "mean", next(iter(point.values.values()))
                )
                for point in result.predictions
            },
        }
        return result
    except ValueError as exc:
        getattr(http_request.app.state, "monitoring", monitoring).record_validation_error(
            route="/predict"
        )
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(payload: FeedbackRequest, request: Request) -> FeedbackResponse:
    predictions = getattr(request.app.state, "predictions", {})
    known_request = predictions.get(payload.request_id)
    if known_request is None:
        raise HTTPException(status_code=404, detail="request_id is unknown or expired")
    if known_request["model_version"] != payload.model_version:
        raise HTTPException(status_code=409, detail="model_version does not match request_id")

    actuals = [
        (point.item_id, point.timestamp.isoformat(), point.sales) for point in payload.actuals
    ]
    metrics = compute_feedback_metrics(known_request["predictions"], actuals)
    if metrics["matched_points"] == 0:
        raise HTTPException(status_code=422, detail="feedback does not match prediction timestamps")

    configured_path = getattr(request.app.state, "feedback_store_path", None)
    store = getattr(request.app.state, "feedback_store", None)
    if store is None or (configured_path is not None and store.path != configured_path):
        store = FeedbackStore(configured_path or Path("/tmp/forecast-api-feedback.jsonl"))
        request.app.state.feedback_store = store
    store.append(
        {
            "request_id": payload.request_id,
            "model_version": payload.model_version,
            "accepted_points": int(metrics["matched_points"]),
            "metrics": metrics,
        }
    )
    getattr(request.app.state, "monitoring", monitoring).record_feedback(
        model_version=payload.model_version,
        status="accepted",
        count=int(metrics["matched_points"]),
    )
    return FeedbackResponse(
        request_id=payload.request_id,
        model_version=payload.model_version,
        accepted_points=int(metrics["matched_points"]),
        metrics=metrics,
    )
