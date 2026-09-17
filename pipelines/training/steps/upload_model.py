from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import boto3
from zenml import step

from pipelines.training.steps.package_model import PackagedModel


@dataclass(frozen=True)
class ModelPublication:
    model_uri: str
    model_version: str
    archive_sha256: str


def build_model_uri(bucket: str, model_name: str, model_version: str) -> str:
    if not bucket or not model_name or not model_version:
        raise ValueError("bucket, model_name, and model_version are required")
    return f"s3://{bucket.strip('/')}/{model_name.strip('/')}/{model_version}/model.tar.gz"


def upload_model(
    packaged: PackagedModel,
    *,
    bucket: str,
    model_name: str,
    endpoint_url: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str = "us-east-1",
) -> ModelPublication:
    model_uri = build_model_uri(bucket, model_name, packaged.model_version)
    key = model_uri.removeprefix(f"s3://{bucket}/")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url or None,
        aws_access_key_id=access_key or None,
        aws_secret_access_key=secret_key or None,
        region_name=region,
    )
    client.upload_file(str(packaged.archive_path), bucket, key)
    return ModelPublication(model_uri, packaged.model_version, packaged.archive_sha256)


def upload_model_archive(
    archive_path: Path,
    *,
    model_version: str,
    archive_sha256: str,
    bucket: str,
    model_name: str,
    endpoint_url: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str = "us-east-1",
) -> ModelPublication:
    model_uri = build_model_uri(bucket, model_name, model_version)
    key = model_uri.removeprefix(f"s3://{bucket}/")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url or None,
        aws_access_key_id=access_key or None,
        aws_secret_access_key=secret_key or None,
        region_name=region,
    )
    client.upload_file(str(archive_path), bucket, key)
    return ModelPublication(model_uri, model_version, archive_sha256)


@step(enable_cache=False)
def upload_model_step(
    archive_path: Path,
    model_version: str,
    archive_sha256: str,
    bucket: str,
    model_name: str,
    endpoint_url: str | None = None,
    access_key: str | None = None,
    secret_key: str | None = None,
    region: str = "us-east-1",
) -> tuple[
    Annotated[str, "model_uri"],
    Annotated[str, "model_version"],
    Annotated[str, "archive_sha256"],
]:
    publication = upload_model_archive(
        archive_path,
        model_version=model_version,
        archive_sha256=archive_sha256,
        bucket=bucket,
        model_name=model_name,
        endpoint_url=endpoint_url,
        access_key=access_key,
        secret_key=secret_key,
        region=region,
    )
    return publication.model_uri, publication.model_version, publication.archive_sha256
