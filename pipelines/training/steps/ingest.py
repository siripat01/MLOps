from __future__ import annotations

import polars as pl
from zenml import step

from mlops_project.data.artifacts import load_feature_dataset


@step
def IngestData(artifact_version: str | None = None) -> pl.DataFrame:
    """Load the feature dataset from ZenML by human-readable artifact version."""
    df = load_feature_dataset(version=artifact_version)
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected Polars DataFrame feature artifact, got {type(df)!r}")
    return df
