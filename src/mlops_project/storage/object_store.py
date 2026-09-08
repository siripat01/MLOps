from __future__ import annotations

from io import BytesIO

import boto3
import polars as pl

from mlops_project.config.settings import DataPipelineSettings


def write_parquet_if_configured(
    df: pl.DataFrame, uri: str | None, settings: DataPipelineSettings
) -> str | None:
    if uri is None:
        return None
    bucket, key = parse_s3_uri(uri)
    body = BytesIO()
    df.write_parquet(body)
    body.seek(0)
    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url or None,
        aws_access_key_id=settings.s3_access_key or None,
        aws_secret_access_key=settings.s3_secret_key or None,
        region_name=settings.s3_region,
    )
    client.upload_fileobj(body, bucket, key)
    return uri


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3 URI, got: {uri}")
    bucket_and_key = uri.removeprefix("s3://")
    bucket, _, key = bucket_and_key.partition("/")
    if not bucket or not key:
        raise ValueError(f"Expected s3://bucket/key URI, got: {uri}")
    return bucket, key
