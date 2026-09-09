from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ModelArtifactMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_name: str
    artifact_version: str | None = None
    model_format: str = "autogluon-timeseries-predictor"
    required_files: list[str]


class BentoBuildMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    bento_tag: str
    build_version: str
    git_sha: str
    model_tag: str
    model_artifact_name: str
    model_artifact_version: str | None = None
    dataset_artifact_version: str | None = None
    quality_metrics: dict[str, float]
    quality_thresholds: dict[str, float]


class ImageBuildMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    bento_tag: str
    image_tag: str
    build_version: str
    git_sha: str
    pushed: bool = False
    registry: str | None = None
