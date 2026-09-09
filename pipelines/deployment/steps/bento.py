from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from autogluon.timeseries import TimeSeriesPredictor
from pydantic import BaseModel, ConfigDict
from zenml import step

from pipelines.deployment.steps.quality_gate import QualityGateResult

BENTO_NAME = "store_sales_forecaster"
CONTEXT_ROOT = Path("build/bento_contexts")
MODEL_DIR_NAME = "model"
REQUIRED_AUTOGUON_FILES = (
    "learner.pkl",
    "models/trainer.pkl",
)

SERVICE_SOURCE = '''from __future__ import annotations

from pathlib import Path
from typing import Any

import bentoml
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

MODEL_PATH = Path(__file__).parent / "model"


@bentoml.service(name="store_sales_forecaster", traffic={"timeout": 120})
class StoreSalesForecastService:
    def __init__(self) -> None:
        self.predictor = TimeSeriesPredictor.load(str(MODEL_PATH))

    @bentoml.api
    def forecast(
        self,
        history: list[dict[str, Any]],
        known_covariates: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        history_df = pd.DataFrame(history)
        data = TimeSeriesDataFrame.from_data_frame(
            history_df,
            id_column="item_id",
            timestamp_column="date",
        )

        covariates = None
        if known_covariates:
            covariates_df = pd.DataFrame(known_covariates)
            covariates = TimeSeriesDataFrame.from_data_frame(
                covariates_df,
                id_column="item_id",
                timestamp_column="date",
            )

        predictions = self.predictor.predict(data, known_covariates=covariates)
        return {
            "predictions": predictions.reset_index().to_dict(orient="records"),
        }
'''


class ModelArtifactMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    artifact_name: str | None = None
    artifact_version: str | None = None
    model_format: str = "autogluon-timeseries-predictor"
    required_files: list[str]


class ReleaseMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    git_sha: str
    build_version: str
    model_artifact_name: str | None = None
    model_artifact_version: str | None = None
    dataset_artifact_version: str | None = None
    quality_metrics: dict[str, float]
    quality_thresholds: dict[str, float]


class BentoBuildMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    bento_tag: str
    build_version: str
    git_sha: str
    model_artifact_name: str | None = None
    model_artifact_version: str | None = None
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


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=capture_output,
        text=True,
    )


def _git_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "nogit"
    return result.stdout.strip()


def make_build_version(git_sha: str, timestamp: int | None = None) -> str:
    return f"{git_sha}-{timestamp or int(time.time())}"


def resolve_image_tag(image_tag_template: str, *, version: str, git_sha: str) -> str:
    if "{" in image_tag_template:
        return image_tag_template.format(version=version, git_sha=git_sha)
    if ":" in image_tag_template.rsplit("/", maxsplit=1)[-1]:
        return image_tag_template
    return f"{image_tag_template}:{version}"


def parse_bento_tag(output: str) -> str:
    tag_line = output.strip().splitlines()[-1]
    if tag_line.startswith("__tag__:"):
        return tag_line.removeprefix("__tag__:")
    return tag_line


def validate_autogluon_predictor(model_path: Path) -> None:
    if not model_path.exists() or not model_path.is_dir():
        raise FileNotFoundError(f"Expected AutoGluon predictor directory at {model_path}")

    missing = [
        required_file
        for required_file in REQUIRED_AUTOGUON_FILES
        if not (model_path / required_file).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"AutoGluon predictor directory is missing required files: {missing}"
        )

    TimeSeriesPredictor.load(str(model_path))


def write_bento_context(build_context: Path, model_path: Path) -> None:
    build_context.mkdir(parents=True, exist_ok=True)
    model_target = build_context / MODEL_DIR_NAME
    if model_target.exists():
        shutil.rmtree(model_target)
    shutil.copytree(model_path, model_target)

    (build_context / "service.py").write_text(SERVICE_SOURCE, encoding="utf-8")
    (build_context / "bentofile.yaml").write_text(
        "\n".join(
            [
                'service: "service:StoreSalesForecastService"',
                'description: "Store sales forecasting API packaged by ZenML."',
                "labels:",
                '  project: "mlops-project"',
                f'  model_family: "{BENTO_NAME}"',
                "include:",
                '  - "service.py"',
                '  - "model/**"',
                "python:",
                "  packages:",
                '    - "autogluon.timeseries==1.6.1"',
                '    - "bentoml>=1.3.20,<2"',
                '    - "pandas>=2.0,<2.4"',
                '    - "torch>=2.8,<2.11"',
                "docker:",
                '  python_version: "3.12"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def _artifact_value(metadata: dict[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if value is None:
        return None
    return str(value)


@step(enable_cache=False)
def validate_model_artifact(
    model_path: Path,
    model_artifact_name: str | None = None,
    model_artifact_version: str | None = None,
) -> ModelArtifactMetadata:
    validate_autogluon_predictor(Path(model_path))
    return ModelArtifactMetadata(
        artifact_name=model_artifact_name,
        artifact_version=model_artifact_version,
        required_files=list(REQUIRED_AUTOGUON_FILES),
    )


@step(enable_cache=False)
def prepare_bento_context(
    model_path: Path,
    model_metadata: ModelArtifactMetadata,
    gate: QualityGateResult,
    dataset_metadata: dict[str, Any] | None = None,
) -> Path:
    if not gate.passed:
        raise RuntimeError("Refusing to prepare Bento context because quality gate failed.")

    source = Path(model_path)
    validate_autogluon_predictor(source)

    git_sha = _git_sha()
    version = make_build_version(git_sha)
    build_context = (CONTEXT_ROOT / version).resolve()
    if build_context.exists():
        shutil.rmtree(build_context)

    write_bento_context(build_context, source)
    dataset_metadata = dataset_metadata or {}
    release_metadata = ReleaseMetadata(
        git_sha=git_sha,
        build_version=version,
        model_artifact_name=model_metadata.artifact_name,
        model_artifact_version=model_metadata.artifact_version,
        dataset_artifact_version=_artifact_value(dataset_metadata, "artifact_version"),
        quality_metrics=gate.metrics,
        quality_thresholds=gate.thresholds,
    )
    (build_context / "release_metadata.json").write_text(
        release_metadata.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return build_context


@step(enable_cache=False)
def build_bento(
    build_context: Path,
    gate: QualityGateResult,
) -> BentoBuildMetadata:
    if not gate.passed:
        raise RuntimeError("Refusing to build Bento because quality gate failed.")

    context_path = Path(build_context)
    release_metadata = ReleaseMetadata.model_validate_json(
        (context_path / "release_metadata.json").read_text(encoding="utf-8")
    )
    result = _run(
        [
            "bentoml",
            "build",
            str(context_path),
            "--version",
            release_metadata.build_version,
            "--label",
            f"git_sha={release_metadata.git_sha}",
            "--label",
            "quality_gate=passed",
            "-o",
            "tag",
        ],
        capture_output=True,
    )
    bento_tag = parse_bento_tag(result.stdout)

    return BentoBuildMetadata(
        bento_tag=bento_tag,
        build_version=release_metadata.build_version,
        git_sha=release_metadata.git_sha,
        model_artifact_name=release_metadata.model_artifact_name,
        model_artifact_version=release_metadata.model_artifact_version,
        quality_metrics=release_metadata.quality_metrics,
        quality_thresholds=release_metadata.quality_thresholds,
    )


@step(enable_cache=False)
def build_container_image(
    bento: BentoBuildMetadata,
    image_tag: str = "mlops-project/store-sales-forecast:{version}",
) -> ImageBuildMetadata:
    resolved_image_tag = resolve_image_tag(
        image_tag,
        version=bento.build_version,
        git_sha=bento.git_sha,
    )
    _run(
        [
            "bentoml",
            "containerize",
            bento.bento_tag,
            "-t",
            resolved_image_tag,
            "--progress",
            "plain",
        ],
    )

    return ImageBuildMetadata(
        bento_tag=bento.bento_tag,
        image_tag=resolved_image_tag,
        build_version=bento.build_version,
        git_sha=bento.git_sha,
    )


@step(enable_cache=False)
def push_container_image(
    image: ImageBuildMetadata,
    push: bool = False,
) -> ImageBuildMetadata:
    if not push:
        return image

    _run(["docker", "push", image.image_tag])
    registry = image.image_tag.split("/", maxsplit=1)[0] if "/" in image.image_tag else None
    return image.model_copy(update={"pushed": True, "registry": registry})
