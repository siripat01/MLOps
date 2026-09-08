from __future__ import annotations

import os
from typing import Any


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
