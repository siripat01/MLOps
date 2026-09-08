from __future__ import annotations

from typing import Any

import polars as pl
from zenml import step

from mlops_project.data.artifacts import get_feature_artifact


@step(enable_cache=False)
def load_dataset(artifact_version: str | None = None) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Load the feature dataset from ZenML by human-readable artifact version."""

    artifact = get_feature_artifact(version=artifact_version)

    metadata = {
        "artifact_name": artifact.name,
        "artifact_version": artifact.version,
    }

    return artifact.load(), metadata
