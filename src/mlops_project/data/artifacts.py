from __future__ import annotations

from typing import Any


def load_feature_dataset(
    artifact_name: str = "store_sales_features",
    version: str | None = None,
) -> Any:
    from zenml.client import Client

    artifact = Client().get_artifact_version(artifact_name, version=version)
    return artifact.load()
