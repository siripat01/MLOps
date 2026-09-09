from __future__ import annotations

from pathlib import Path

from zenml import step
from zenml.client import Client


@step(enable_cache=False)
def load_model_artifact(
    artifact_name: str,
    artifact_version: str | None = None,
) -> Path:
    artifact = Client().get_artifact_version(
        name_id_or_prefix=artifact_name,
        version=artifact_version,
    )
    loaded = artifact.load()
    return Path(loaded)
