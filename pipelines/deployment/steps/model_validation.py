from __future__ import annotations

from pathlib import Path

from autogluon.timeseries import TimeSeriesPredictor
from zenml import step

from pipelines.deployment.models import ModelArtifactMetadata

REQUIRED_AUTOGLOON_FILES = (
    "learner.pkl",
    "models/trainer.pkl",
)


def validate_autogluon_predictor(model_path: Path) -> None:
    if not model_path.is_dir():
        raise FileNotFoundError(f"Expected AutoGluon predictor directory at {model_path}")

    missing = [
        relative_path
        for relative_path in REQUIRED_AUTOGLOON_FILES
        if not (model_path / relative_path).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"AutoGluon predictor directory is missing required files: {missing}"
        )

    TimeSeriesPredictor.load(str(model_path))


@step(enable_cache=False)
def validate_model_artifact(
    model_path: Path,
    model_artifact_name: str,
    model_artifact_version: str | None = None,
) -> ModelArtifactMetadata:
    validate_autogluon_predictor(Path(model_path))
    return ModelArtifactMetadata(
        artifact_name=model_artifact_name,
        artifact_version=model_artifact_version,
        required_files=list(REQUIRED_AUTOGLOON_FILES),
    )
