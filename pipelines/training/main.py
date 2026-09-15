from __future__ import annotations

import os

from zenml import pipeline
from zenml.config import DockerSettings
from zenml.orchestrators.local_docker.local_docker_orchestrator import (
    LocalDockerOrchestratorSettings,
)

from pipelines.training.steps.eval import evaluate_model
from pipelines.training.steps.load_feature import load_dataset
from pipelines.training.steps.prepare_split import prepare_training_data
from pipelines.training.steps.split_data import split_data
from pipelines.training.steps.training import train_model


def _docker_feature_uri() -> str:
    feature_version = (
        os.getenv("FEATURE_DATA_VERSION")
        or os.getenv("DATASET_VERSION")
        or "local-dev"
    )
    return os.getenv("TRAIN_FEATURE_URI") or (
        f"s3://{os.getenv('S3_BUCKET', 'zenml')}/"
        f"{os.getenv('FEATURE_PREFIX', 'features')}/"
        f"{os.getenv('DATASET_NAME', 'store-sales')}/"
        f"{feature_version}/features.parquet"
    )


def _docker_endpoint(env_name: str, default: str) -> str:
    value = os.getenv(env_name) or default
    return value.replace("localhost", "172.17.0.1")


docker = DockerSettings(
    parent_image="pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime",
    python_package_installer="uv",
    python_package_installer_args={
        "system": None,
    },
    requirements=[
        "zenml==0.96.4",
        "autogluon.timeseries==1.6.1",
        "mlflow>=2.1.1,<4",
        "torch==2.13.0",
        "torchvision==0.28.0",
        "numpy",
        "pandas",
        "polars",
        "pyarrow",
        "s3fs",
    ],
    target_repository="zenml",
    prevent_build_reuse=False,
    local_project_install_command="pip install --no-deps -e .",
    environment={
        "PYTHONPATH": "/app/code/src:/app/code",
        "MPLCONFIGDIR": "/tmp/matplotlib",
        "TRAIN_FEATURE_URI": _docker_feature_uri(),
        "S3_ENDPOINT_URL": _docker_endpoint("S3_ENDPOINT_URL", "http://172.17.0.1:9000"),
        "S3_ACCESS_KEY": os.getenv("S3_ACCESS_KEY", "minioadmin"),
        "S3_SECRET_KEY": os.getenv("S3_SECRET_KEY", "local-dev-minio"),
        "S3_BUCKET": os.getenv("S3_BUCKET", "zenml"),
        "S3_REGION": os.getenv("S3_REGION", "us-east-1"),
        "MLFLOW_TRACKING_URI": _docker_endpoint(
            "MLFLOW_DOCKER_TRACKING_URI",
            "http://172.17.0.1:5000",
        ),
    },
)

orchestrator_settings = LocalDockerOrchestratorSettings(
    run_args={
        "runtime": "nvidia",
        "cpu_count": 20,
        "mem_limit": "16g",
        "shm_size": "2g",
        "environment": {
            "NVIDIA_VISIBLE_DEVICES": "all",
        },
    }
)


@pipeline(
    settings={
        "docker": docker,
        "orchestrator.local_docker": orchestrator_settings,
    }
)
def training_pipeline(
    artifact_version: str | None = None,
    prediction_length: int = 16,
    presets: str = "high_quality",
    eval_metric: str = "RMSLE",
    time_limit: int | None = None,
    enable_ensemble: bool = True,
    model_profile: str | None = "local_safe",
) -> None:
    dataset, _artifact_metadata = load_dataset(artifact_version=artifact_version)
    train_df, validate_df, _spliting_metadata = split_data(
        df=dataset,
        dataset_metadata=_artifact_metadata,
        prediction_length=prediction_length,
    )

    train_ts = prepare_training_data(train_df)
    validation_ts = prepare_training_data(validate_df)

    model_path, training_metadata = train_model(
        train_ts,
        prediction_length=prediction_length,
        presets=presets,
        eval_metric=eval_metric,
        time_limit=time_limit,
        enable_ensemble=enable_ensemble,
        model_profile=model_profile,
    )

    evaluate_model(
        model_path,
        train_ts,
        validation_ts,
        training_metadata,
    )


def main() -> None:
    training_pipeline(
        artifact_version=os.getenv("TRAIN_FEATURE_VERSION") or None,
        prediction_length=int(os.getenv("PREDICTION_LENGTH", "16")),
        presets=os.getenv("AUTOGLUON_PRESETS", "high_quality"),
        eval_metric=os.getenv("AUTOGLUON_EVAL_METRIC", "RMSLE"),
        time_limit=(
            int(os.environ["AUTOGLUON_TIME_LIMIT"]) if os.getenv("AUTOGLUON_TIME_LIMIT") else None
        ),
        enable_ensemble=os.getenv("AUTOGLUON_ENABLE_ENSEMBLE", "true").lower()
        not in {"0", "false", "no"},
        model_profile=os.getenv("AUTOGLUON_MODEL_PROFILE", "local_safe"),
    )


if __name__ == "__main__":
    main()
