import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app
from app.model import validate_prediction_request
from app.schemas import ModelMetadata, PredictionRequest


def payload() -> dict:
    return {
        "history": [{"item_id": "1", "date": "2024-01-01", "sales": 10}],
        "known_covariates": [{"item_id": "1", "date": "2024-01-02"}],
    }


def test_health_does_not_require_model() -> None:
    app.state.__dict__.pop("model", None)
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_fails_before_model_load() -> None:
    app.state.__dict__.pop("model", None)
    response = TestClient(app, raise_server_exceptions=False).get("/ready")
    assert response.status_code == 503


def test_predict_rejects_unknown_fields() -> None:
    app.state.model = object()
    response = TestClient(app).post("/predict", json={**payload(), "extra": True})
    assert response.status_code == 422


def test_predict_uses_loaded_model() -> None:
    class FakeModel:
        def predict(self, request):
            return {"predictions": []}

    app.state.model = FakeModel()
    response = TestClient(app).post("/predict", json=payload())
    assert response.status_code == 200
    assert response.json() == {"predictions": []}


def test_predict_maps_model_input_errors_to_422() -> None:
    class RejectingModel:
        def predict(self, request):
            raise ValueError("known_covariates horizon is invalid")

    app.state.model = RejectingModel()
    response = TestClient(app).post("/predict", json=payload())

    assert response.status_code == 422
    assert response.json()["detail"] == "known_covariates horizon is invalid"


def test_prediction_validation_requires_full_future_horizon() -> None:
    request = PredictionRequest(**payload())

    with pytest.raises(ValueError, match="16 future"):
        validate_prediction_request(request, prediction_length=16)


def test_prediction_validation_rejects_mismatched_items() -> None:
    request = PredictionRequest(
        history=[{"item_id": "1", "date": "2024-01-01", "sales": 10}],
        known_covariates=[{"item_id": "2", "date": "2024-01-02"}],
    )

    with pytest.raises(ValueError, match="item_id"):
        validate_prediction_request(request, prediction_length=1)


def test_lifespan_loads_model_before_ready(monkeypatch, tmp_path) -> None:
    metadata = ModelMetadata(
        model_name="store-sales",
        model_version="v19",
        artifact_uri="s3://bucket/model.tar.gz",
        framework="autogluon-timeseries",
        autogluon_version="1.6.1",
        python_version="3.12.0",
        artifact_sha256="a" * 64,
    )

    class FakeDownloader:
        def __init__(self, *_args, **_kwargs):
            pass

        def download_and_extract(self, *_args, **_kwargs):
            return tmp_path, metadata.model_dump()

    class FakeModel:
        @classmethod
        def load(cls, *_args, **_kwargs):
            return cls()

    monkeypatch.setattr(
        "app.main.Settings.from_env",
        lambda: Settings("s3://bucket/model.tar.gz", None, tmp_path, None, None, None, "us-east-1"),
    )
    monkeypatch.setattr("app.main.ArtifactDownloader", FakeDownloader)
    monkeypatch.setattr("app.main.AutoGluonModel", FakeModel)

    with TestClient(app) as client:
        assert client.get("/ready").status_code == 200
        assert client.get("/metadata").json()["model_version"] == "v19"


def test_lifespan_surfaces_model_load_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "app.main.Settings.from_env",
        lambda: Settings("s3://bucket/model.tar.gz", None, tmp_path, None, None, None, "us-east-1"),
    )

    class FakeDownloader:
        def __init__(self, *_args, **_kwargs):
            pass

        def download_and_extract(self, *_args, **_kwargs):
            return tmp_path, {
                "model_name": "store-sales",
                "model_version": "v19",
                "framework": "autogluon-timeseries",
                "autogluon_version": "1.6.1",
                "python_version": "3.12.0",
                "artifact_sha256": "a" * 64,
            }

    class FailingModel:
        @classmethod
        def load(cls, *_args, **_kwargs):
            raise RuntimeError("predictor is incompatible")

    monkeypatch.setattr("app.main.ArtifactDownloader", FakeDownloader)
    monkeypatch.setattr("app.main.AutoGluonModel", FailingModel)
    with pytest.raises(RuntimeError, match="incompatible"):
        with TestClient(app):
            pass
