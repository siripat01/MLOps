import os
from typing import Annotated, Any, Tuple  # noqa: UP035

import polars as pl
from zenml import step

from mlops_project.data.artifacts import (
    feature_uri_from_environment,
    get_feature_artifact,
    load_feature_dataset_from_uri,
)


def resolve_feature_dataset(
    artifact_version: str | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    try:
        artifact = get_feature_artifact(version=artifact_version)
    except Exception as exc:
        feature_uri = feature_uri_from_environment()
        if not feature_uri:
            requested = artifact_version or "latest"
            artifact_name = os.getenv("FEATURE_ARTIFACT_NAME", "store_sales_features")
            raise KeyError(
                "Unable to load feature dataset artifact "
                f"'{artifact_name}' version '{requested}'. "
                "Run the data pipeline first, set TRAIN_FEATURE_VERSION to an "
                "existing version, or provide TRAIN_FEATURE_URI for direct Parquet "
                "fallback."
            ) from exc
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


@step(enable_cache=False)
def load_dataset(
    artifact_version: str | None = None,
) -> Tuple[  # noqa: UP006
    Annotated[pl.DataFrame, "feature_dataset"],
    Annotated[dict[str, Any], "feature_dataset_metadata"],
]:
    """Load feature data from ZenML, or direct URI when metadata is unavailable."""

    dataset, metadata = resolve_feature_dataset(artifact_version=artifact_version)
    return dataset, metadata
