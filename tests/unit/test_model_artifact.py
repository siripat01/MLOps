import hashlib
import json
import tarfile
from pathlib import Path
from typing import get_args, get_type_hints

import pytest

from pipelines.training.steps.package_model import package_model, package_model_step
from pipelines.training.steps.quality_gate import quality_gate
from pipelines.training.steps.upload_model import (
    build_model_uri,
    upload_model,
    upload_model_step,
)


def predictor_dir(root: Path) -> Path:
    model = root / "predictor"
    (model / "models").mkdir(parents=True)
    (model / "predictor.pkl").write_bytes(b"predictor")
    (model / "learner.pkl").write_bytes(b"learner")
    (model / "models" / "trainer.pkl").write_bytes(b"trainer")
    return model


def test_package_model_contains_complete_predictor_and_manifest(tmp_path: Path) -> None:
    packaged = package_model(
        predictor_dir(tmp_path),
        model_name="store-sales",
        model_version="v19",
        artifact_root=tmp_path / "artifacts",
    )

    with tarfile.open(packaged.archive_path, "r:gz") as archive:
        names = set(archive.getnames())
        assert {
            "model/predictor.pkl",
            "model/learner.pkl",
            "model/models/trainer.pkl",
            "manifest.json",
        } <= names
        manifest = json.loads(archive.extractfile("manifest.json").read())

    assert manifest["model_name"] == "store-sales"
    assert manifest["model_version"] == "v19"
    assert manifest["framework"] == "autogluon-timeseries"
    assert manifest["artifact_sha256"] == packaged.model_sha256
    assert len(packaged.archive_sha256) == 64


def test_model_uri_is_versioned() -> None:
    assert build_model_uri("ml-models", "store-sales", "v19") == (
        "s3://ml-models/store-sales/v19/model.tar.gz"
    )


def test_package_step_exposes_portable_archive_outputs() -> None:
    annotations = get_type_hints(package_model_step.entrypoint, include_extras=True)["return"]

    assert len(get_args(annotations)) == 3


def test_upload_model_uses_s3_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packaged = package_model(
        predictor_dir(tmp_path),
        model_name="store-sales",
        model_version="v19",
        artifact_root=tmp_path / "artifacts",
    )
    calls: list[tuple[str, str, str]] = []

    class FakeClient:
        def upload_file(self, filename: str, bucket: str, key: str) -> None:
            calls.append((filename, bucket, key))

    monkeypatch.setattr(
        "pipelines.training.steps.upload_model.boto3.client",
        lambda *_args, **_kwargs: FakeClient(),
    )

    publication = upload_model(
        packaged,
        bucket="ml-models",
        model_name="store-sales",
        endpoint_url="http://minio:9000",
    )

    assert publication.model_uri == "s3://ml-models/store-sales/v19/model.tar.gz"
    assert calls == [(str(packaged.archive_path), "ml-models", "store-sales/v19/model.tar.gz")]
    assert hashlib.sha256(packaged.archive_path.read_bytes()).hexdigest() == packaged.archive_sha256


def test_upload_step_exposes_zenml_compatible_named_outputs() -> None:
    annotations = get_type_hints(upload_model_step.entrypoint, include_extras=True)["return"]
    assert len(get_args(annotations)) == 3


def test_quality_gate_exposes_zenml_compatible_named_outputs() -> None:
    annotations = get_type_hints(quality_gate.entrypoint, include_extras=True)["return"]
    assert len(get_args(annotations)) == 3
