from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

import boto3


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3://bucket/key URI, got: {uri}")
    bucket, separator, key = uri.removeprefix("s3://").partition("/")
    if not bucket or not separator or not key:
        raise ValueError(f"Expected s3://bucket/key URI, got: {uri}")
    return bucket, key


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(child.read_bytes())
    return digest.hexdigest()


class ArtifactDownloader:
    def __init__(
        self,
        cache_dir: Path,
        *,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str = "us-east-1",
    ) -> None:
        self.cache_dir = cache_dir
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint_url or None,
            aws_access_key_id=access_key or None,
            aws_secret_access_key=secret_key or None,
            region_name=region,
        )

    def download_and_extract(
        self, uri: str, expected_sha256: str | None = None
    ) -> tuple[Path, dict[str, Any]]:
        bucket, key = parse_s3_uri(uri)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        archive_path = self.cache_dir / "model.tar.gz"
        model_dir = self.cache_dir / "model"
        manifest_path = self.cache_dir / "manifest.json"
        marker_path = self.cache_dir / "cache-key.json"

        cache_key = {"uri": uri, "expected_sha256": expected_sha256}
        cached_key = None
        if marker_path.is_file():
            try:
                cached_key = json.loads(marker_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cached_key = None
        cache_matches = cached_key == cache_key

        if not cache_matches or not archive_path.exists() or (
            expected_sha256 and sha256_file(archive_path) != expected_sha256
        ):
            if model_dir.exists():
                shutil.rmtree(model_dir)
            manifest_path.unlink(missing_ok=True)
            self.client.download_file(bucket, key, str(archive_path))
        if expected_sha256 and sha256_file(archive_path) != expected_sha256:
            raise ValueError("Downloaded model artifact checksum does not match MODEL_SHA256")

        if not model_dir.is_dir() or not manifest_path.is_file():
            if model_dir.exists():
                shutil.rmtree(model_dir)
            self._extract_safely(archive_path, self.cache_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError("Artifact does not contain a model/ directory")
        if not manifest_path.is_file():
            raise FileNotFoundError("Artifact does not contain manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_model_sha256 = manifest.get("artifact_sha256")
        if expected_model_sha256 and sha256_directory(model_dir) != expected_model_sha256:
            raise ValueError("Extracted model checksum does not match manifest.json")
        marker_path.write_text(json.dumps(cache_key, sort_keys=True), encoding="utf-8")
        return model_dir, manifest

    @staticmethod
    def _extract_safely(archive_path: Path, destination: Path) -> None:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                target = PurePosixPath(member.name)
                if target.is_absolute() or ".." in target.parts:
                    raise ValueError(f"Unsafe path in model archive: {member.name}")
            archive.extractall(destination, filter="data")
