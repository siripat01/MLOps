from __future__ import annotations

import os
from typing import Any

import polars as pl

from mlops_project.config.settings import get_settings


def normalize_artifact_version(version: str | int | None) -> str | None:
    """Accept human-friendly aliases such as v1 while using ZenML auto versions."""
    if version is None:
        return None
    value = str(version).strip()
    if not value or value.lower() == "latest":
        return None
    if value.lower().startswith("v") and value[1:].isdigit():
        return value[1:]
    return value


def get_feature_artifact(
    artifact_name: str | None = None,
    version: str | int | None = None,
) -> Any:
    from zenml.client import Client

    name = artifact_name or os.getenv("FEATURE_ARTIFACT_NAME", "store_sales_features")
    normalized_version = normalize_artifact_version(version)
    return Client().get_artifact_version(
        name_id_or_prefix=name,
        version=normalized_version,
    )


def load_feature_dataset(
    artifact_name: str | None = None,
    version: str | int | None = None,
) -> Any:
    return get_feature_artifact(artifact_name=artifact_name, version=version).load()


def feature_uri_from_environment() -> str | None:
    """Return the direct feature URI used when ZenML artifact metadata is absent."""
    explicit_uri = os.getenv("TRAIN_FEATURE_URI")
    if explicit_uri:
        return explicit_uri
    return get_settings().feature_uri


def load_feature_dataset_from_uri(uri: str) -> pl.DataFrame:
    settings = get_settings()
    storage_options: dict[str, str] = {}

    if uri.startswith("s3://"):
        if settings.s3_access_key:
            storage_options["aws_access_key_id"] = settings.s3_access_key
        if settings.s3_secret_key:
            storage_options["aws_secret_access_key"] = settings.s3_secret_key
        if settings.s3_region:
            storage_options["aws_region"] = settings.s3_region
        if settings.s3_endpoint_url:
            storage_options["aws_endpoint_url"] = settings.s3_endpoint_url
            if settings.s3_endpoint_url.startswith("http://"):
                storage_options["aws_allow_http"] = "true"

    return pl.read_parquet(uri, storage_options=storage_options or None)
