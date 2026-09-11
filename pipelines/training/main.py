from __future__ import annotations

import os

from dotenv import load_dotenv
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

load_dotenv()

# Docker step containers cannot reach the host through ``localhost``. Keep the
# endpoint configurable while using the default Docker bridge gateway locally.
os.environ.setdefault(
    "ZENML_STORE_URL",
    os.getenv("ZENML_DOCKER_STORE_URL", "http://172.17.0.1:8080"),
)

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
        "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_DOCKER_TRACKING_URI", "http://172.17.0.1:5000"),
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
    presets: str = "best_quality",
    eval_metric: str = "RMSLE",
    time_limit: int | None = None,
    enable_ensemble: bool = True,
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
        presets=os.getenv("AUTOGLUON_PRESETS", "best_quality"),
        eval_metric=os.getenv("AUTOGLUON_EVAL_METRIC", "RMSLE"),
        time_limit=(
            int(os.environ["AUTOGLUON_TIME_LIMIT"])
            if os.getenv("AUTOGLUON_TIME_LIMIT")
            else None
        ),
        enable_ensemble=os.getenv("AUTOGLUON_ENABLE_ENSEMBLE", "true").lower()
        not in {"0", "false", "no"},
    )


if __name__ == "__main__":
    main()
