from __future__ import annotations

import os

from dotenv import load_dotenv
from zenml import pipeline
from zenml.config import DockerSettings

from pipelines.deployment.steps.bento import build_bento
from pipelines.deployment.steps.image import build_container_image, push_container_image
from pipelines.deployment.steps.model_artifact import load_model_artifact
from pipelines.deployment.steps.model_validation import validate_model_artifact
from pipelines.deployment.steps.quality_gate import quality_gate
from pipelines.training.steps.eval import evaluate_model
from pipelines.training.steps.load_feature import load_dataset
from pipelines.training.steps.prepare_split import prepare_training_data
from pipelines.training.steps.split_data import split_data

load_dotenv()

docker = DockerSettings(
    parent_image="python:3.12-slim",
    python_package_installer="uv",
    python_package_installer_args={"system": None},
    requirements=[
        "zenml==0.96.4",
        "autogluon.timeseries==1.6.1",
        "bentoml>=1.3.20,<2",
        "pandas>=2.0,<2.4",
        "polars>=1.0,<2",
        "pyarrow",
        "python-dotenv>=1,<2",
        "s3fs",
    ],
    target_repository="zenml",
    prevent_build_reuse=False,
    local_project_install_command="pip install --no-deps -e .",
    environment={
        "PYTHONPATH": "/app/code/src:/app/code",
        "KAGGLEHUB_CACHE": "/tmp/kagglehub",
    },
)


@pipeline(settings={"docker": docker})
def bento_image_build_pipeline(
    model_artifact_name: str,
    model_artifact_version: str | None = None,
    artifact_version: str | None = None,
    prediction_length: int = 16,
    max_rmsle: float = 0.75,
    max_wql: float | None = None,
    max_rmse: float | None = None,
    image_tag: str = "mlops-project/store-sales-forecast:{version}",
    push_image: bool = False,
) -> None:
    """Validate and release an existing model artifact as a Bento OCI image."""
    dataset, artifact_metadata = load_dataset(artifact_version=artifact_version)
    train_df, validate_df, _split_metadata = split_data(
        df=dataset,
        dataset_metadata=artifact_metadata,
        prediction_length=prediction_length,
    )

    train_ts = prepare_training_data(train_df)
    validation_ts = prepare_training_data(validate_df)

    model_path = load_model_artifact(model_artifact_name, model_artifact_version)
    metrics = evaluate_model(model_path, train_ts, validation_ts)

    gate = quality_gate(
        metrics=metrics,
        max_rmsle=max_rmsle,
        max_wql=max_wql,
        max_rmse=max_rmse,
    )
    model_metadata = validate_model_artifact(
        model_path,
        model_artifact_name=model_artifact_name,
        model_artifact_version=model_artifact_version,
    )
    bento_archive, bento_metadata = build_bento(
        model_path,
        model_metadata,
        gate,
        artifact_metadata,
    )
    image = build_container_image(
        bento_archive,
        bento_metadata,
        image_tag=image_tag,
    )
    push_container_image(image, push=push_image)


def _optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return None if value in (None, "") else float(value)


def main() -> None:
    model_artifact_name = os.getenv("MODEL_ARTIFACT_NAME", "store_sales_model").strip()
    if not model_artifact_name:
        raise RuntimeError("MODEL_ARTIFACT_NAME must identify a trained ZenML model artifact")

    bento_image_build_pipeline(
        artifact_version=os.getenv("TRAIN_FEATURE_VERSION") or None,
        model_artifact_name=model_artifact_name,
        model_artifact_version=os.getenv("MODEL_ARTIFACT_VERSION") or None,
        prediction_length=int(os.getenv("PREDICTION_LENGTH", "16")),
        max_rmsle=float(os.getenv("QUALITY_GATE_MAX_RMSLE", "0.75")),
        max_wql=_optional_float("QUALITY_GATE_MAX_WQL"),
        max_rmse=_optional_float("QUALITY_GATE_MAX_RMSE"),
        image_tag=os.getenv(
            "BENTO_IMAGE_TAG",
            "mlops-project/store-sales-forecast:{version}",
        ),
        push_image=os.getenv("BENTO_PUSH_IMAGE", "false").lower()
        in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    main()
