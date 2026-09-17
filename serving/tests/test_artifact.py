import json
import tarfile
from pathlib import Path

import pytest

from app.artifact import ArtifactDownloader, parse_s3_uri


def test_parse_s3_uri() -> None:
    assert parse_s3_uri("s3://bucket/path/model.tar.gz") == ("bucket", "path/model.tar.gz")


def test_parse_s3_uri_rejects_invalid_uri() -> None:
    with pytest.raises(ValueError):
        parse_s3_uri("https://bucket/path")


def test_downloader_downloads_extracts_and_reads_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    (source / "model").mkdir(parents=True)
    (source / "model" / "learner.pkl").write_bytes(b"model")
    (source / "manifest.json").write_text(
        json.dumps({"model_name": "store-sales"}), encoding="utf-8"
    )
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source / "model", arcname="model")
        tar.add(source / "manifest.json", arcname="manifest.json")

    class FakeClient:
        def download_file(self, bucket: str, key: str, filename: str) -> None:
            Path(filename).write_bytes(archive.read_bytes())

    monkeypatch.setattr("app.artifact.boto3.client", lambda *_args, **_kwargs: FakeClient())
    model_dir, manifest = ArtifactDownloader(tmp_path / "cache").download_and_extract(
        "s3://bucket/model.tar.gz"
    )

    assert (model_dir / "learner.pkl").read_bytes() == b"model"
    assert manifest == {"model_name": "store-sales"}


def test_downloader_rejects_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeClient:
        def download_file(self, bucket: str, key: str, filename: str) -> None:
            Path(filename).write_bytes(b"corrupt")

    monkeypatch.setattr("app.artifact.boto3.client", lambda *_args, **_kwargs: FakeClient())
    with pytest.raises(ValueError, match="checksum"):
        ArtifactDownloader(tmp_path / "cache").download_and_extract(
            "s3://bucket/model.tar.gz", expected_sha256="0" * 64
        )


def test_downloader_rejects_corrupt_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeClient:
        def download_file(self, bucket: str, key: str, filename: str) -> None:
            Path(filename).write_bytes(b"not a gzip archive")

    monkeypatch.setattr("app.artifact.boto3.client", lambda *_args, **_kwargs: FakeClient())
    with pytest.raises(tarfile.ReadError):
        ArtifactDownloader(tmp_path / "cache").download_and_extract("s3://bucket/model.tar.gz")
