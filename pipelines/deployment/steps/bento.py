from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Annotated, Any

import bentoml
from dotenv import load_dotenv
from zenml import ArtifactConfig, step
from zenml.enums import ArtifactType

from pipelines.deployment.models import BentoBuildMetadata, ModelArtifactMetadata
from pipelines.deployment.steps.model_validation import validate_autogluon_predictor
from pipelines.deployment.steps.quality_gate import QualityGateResult

load_dotenv()

_DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]

BENTO_NAME = os.getenv("BENTO_NAME", "store_sales_forecaster")
BENTO_MODEL_NAME = os.getenv("BENTO_MODEL_NAME", "store_sales_forecaster_model")
BENTO_MODEL_ALIAS = os.getenv("BENTO_MODEL_ALIAS", "store_sales_model")
SERVICE_IMPORT = os.getenv(
    "SERVICE_IMPORT",
    "mlops_project.serving.service:StoreSalesForecastService",
)
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(_DEFAULT_PROJECT_ROOT))).resolve()
SOURCE_ROOT = Path(os.getenv("SOURCE_ROOT", "src"))
if not SOURCE_ROOT.is_absolute():
    SOURCE_ROOT = PROJECT_ROOT / SOURCE_ROOT
BENTO_EXPORT_ROOT = Path(os.getenv("BENTO_EXPORT_ROOT", "build/bentos"))
if not BENTO_EXPORT_ROOT.is_absolute():
    BENTO_EXPORT_ROOT = PROJECT_ROOT / BENTO_EXPORT_ROOT


def _git_sha() -> str:
    """Read the source revision injected by the release environment."""
    git_sha = os.getenv("GIT_SHA") or os.getenv("GITHUB_SHA")
    if not git_sha:
        return "nogit"
    return git_sha[:12]


def make_build_version(git_sha: str, timestamp: int | None = None) -> str:
    return f"{git_sha}-{timestamp or int(time.time())}"


def _artifact_value(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return None if value is None else str(value)


def _save_model_to_bentoml(
    model_path: Path,
    model_metadata: ModelArtifactMetadata,
) -> str:
    labels = {
        "project": "mlops-project",
        "source_artifact": model_metadata.artifact_name,
    }
    metadata: dict[str, str] = {
        "source_artifact_name": model_metadata.artifact_name,
    }
    if model_metadata.artifact_version:
        metadata["source_artifact_version"] = model_metadata.artifact_version

    with bentoml.models.create(
        BENTO_MODEL_NAME,
        labels=labels,
        metadata=metadata,
    ) as model_ref:
        shutil.copytree(model_path, model_ref.path, dirs_exist_ok=True)
        return str(model_ref.tag)


@step(enable_cache=False)
def build_bento(
    model_path: Path,
    model_metadata: ModelArtifactMetadata,
    gate: QualityGateResult,
    dataset_metadata: dict[str, Any] | None = None,
) -> tuple[
    Annotated[
        Path,
        ArtifactConfig(name="store_sales_bento", artifact_type=ArtifactType.MODEL),
    ],
    BentoBuildMetadata,
]:
    """Build and export a self-contained Bento bundle.

    The exported `.bento` file is returned as a ZenML Path artifact so downstream
    steps do not depend on a process-local Bento store.
    """
    if not gate.passed:
        raise RuntimeError("Refusing to build Bento because quality gate failed.")

    source = Path(model_path)
    validate_autogluon_predictor(source)

    git_sha = _git_sha()
    build_version = make_build_version(git_sha)
    dataset_metadata = dataset_metadata or {}

    model_tag = _save_model_to_bentoml(source, model_metadata)
    bento = bentoml.build(
        service=SERVICE_IMPORT,
        name=BENTO_NAME,
        version=build_version,
        build_ctx=str(SOURCE_ROOT),
        models=[
            {
                "tag": model_tag,
                "alias": BENTO_MODEL_ALIAS,
            }
        ],
        labels={
            "project": "mlops-project",
            "git_sha": git_sha,
            "quality_gate": "passed",
        },
    )

    BENTO_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    export_path = BENTO_EXPORT_ROOT / f"{build_version}.bento"
    if export_path.exists():
        export_path.unlink()

    exported = Path(bentoml.export_bento(str(bento.tag), str(export_path))).resolve()

    metadata = BentoBuildMetadata(
        bento_tag=str(bento.tag),
        build_version=build_version,
        git_sha=git_sha,
        model_tag=model_tag,
        model_artifact_name=model_metadata.artifact_name,
        model_artifact_version=model_metadata.artifact_version,
        dataset_artifact_version=_artifact_value(dataset_metadata, "artifact_version"),
        quality_metrics=gate.metrics,
        quality_thresholds=gate.thresholds,
    )
    return exported, metadata
