from __future__ import annotations

import os
import time

from zenml import pipeline
from zenml.config import DockerSettings
from zenml.orchestrators.local_docker.local_docker_orchestrator import (
    LocalDockerOrchestratorSettings,
)

from pipelines.training.steps.eval import evaluate_model
from pipelines.training.steps.load_feature import load_dataset
from pipelines.training.steps.package_model import package_model_step
from pipelines.training.steps.prepare_split import prepare_training_data
from pipelines.training.steps.quality_gate import quality_gate
from pipelines.training.steps.split_data import split_data
from pipelines.training.steps.training import train_model
from pipelines.training.steps.upload_model import upload_model_step

TRAINING_RUNNER_IMAGE = os.getenv(
    "TRAINING_RUNNER_IMAGE",
    "docker.io/siripat007/zenml:training-runner-autogluon-1.6.1-torch2.10",
)


def _docker_feature_uri() -> str:
    feature_version = (
        os.getenv("FEATURE_DATA_VERSION") or os.getenv("DATASET_VERSION") or "local-dev"
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


def _optional_float_env(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    return float(value) if value else None


docker = DockerSettings(
    parent_image=TRAINING_RUNNER_IMAGE,
    # The runner image already contains all dependencies. Using pip here avoids
    # ZenML bootstrapping uv with a failing `pip install uv` layer.
    python_package_installer="pip",
    install_stack_requirements=False,
    # The runner image already contains all runtime dependencies. Keeping this
    # empty prevents every ZenML step image build from downloading AutoGluon.
    requirements=[],
    target_repository="zenml",
    prevent_build_reuse=False,
    local_project_install_command=(
        "uv pip install --system --break-system-packages --no-deps -e ."
    ),
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
    max_wql: float | None = None,
    max_rmse: float | None = None,
    model_name: str = "store-sales",
    model_version: str | None = None,
) -> tuple[str, str, dict[str, float]]:
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

    metrics = evaluate_model(
        model_path,
        train_ts,
        validation_ts,
        training_metadata,
    )
    gate_passed, gate_metrics, _gate_thresholds = quality_gate(
        metrics=metrics,
        max_rmsle=float(os.getenv("QUALITY_GATE_MAX_RMSLE", "0.75")),
        max_wql=max_wql,
        max_rmse=max_rmse,
    )
    archive_path, packaged_model_version, archive_sha256 = package_model_step(
        model_path,
        model_name=model_name,
        model_version=model_version or f"v{int(time.time())}",
        artifact_root=os.getenv("MODEL_PACKAGE_ROOT", "/tmp/model-publication"),
        quality_gate_passed=gate_passed,
        prediction_length=prediction_length,
    )
    model_uri, published_model_version, _archive_sha256 = upload_model_step(
        archive_path,
        model_version=packaged_model_version,
        archive_sha256=archive_sha256,
        bucket=os.getenv("MODEL_BUCKET", os.getenv("S3_BUCKET", "ml-models")),
        model_name=model_name,
        endpoint_url=os.getenv("S3_ENDPOINT_URL"),
        access_key=os.getenv("S3_ACCESS_KEY"),
        secret_key=os.getenv("S3_SECRET_KEY"),
        region=os.getenv("S3_REGION", "us-east-1"),
    )
    return model_uri, published_model_version, gate_metrics


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
        max_wql=_optional_float_env("QUALITY_GATE_MAX_WQL"),
        max_rmse=_optional_float_env("QUALITY_GATE_MAX_RMSE"),
        model_name=os.getenv("MODEL_NAME", "store-sales"),
        model_version=os.getenv("MODEL_VERSION") or None,
    )


if __name__ == "__main__":
    main()
