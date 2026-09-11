from __future__ import annotations

from typing import Any

import polars as pl
from zenml import step

from mlops_project.data.artifacts import (
    feature_uri_from_environment,
    get_feature_artifact,
    load_feature_dataset_from_uri,
)


@step(enable_cache=False)
def load_dataset(artifact_version: str | None = None) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Load the feature dataset from ZenML by human-readable artifact version."""

    try:
        artifact = get_feature_artifact(version=artifact_version)
    except Exception as exc:
        feature_uri = feature_uri_from_environment()
        if not feature_uri:
            raise
        metadata = {
            "source": "direct_uri",
            "feature_uri": feature_uri,
            "requested_artifact_version": artifact_version,
            "artifact_lookup_error": str(exc),
        }
        return load_feature_dataset_from_uri(feature_uri), metadata

    metadata = {
        "source": "zenml_artifact",
        "artifact_name": artifact.name,
        "artifact_version": artifact.version,
    }
    return artifact.load(), metadata
