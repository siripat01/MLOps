from __future__ import annotations

import os

from zenml import pipeline
from zenml.config import DockerSettings
from zenml.orchestrators.local_docker.local_docker_orchestrator import (
    LocalDockerOrchestratorSettings,
)

from pipelines.data.steps.clean import clean_data
from pipelines.data.steps.feature_engineering import engineer_features
from pipelines.data.steps.ingest import ingest_data
from pipelines.data.steps.integrate import integrate_data
from pipelines.data.steps.profile import profile_data
from pipelines.data.steps.transform import transform_data
from pipelines.data.steps.validate_cleaned import validate_cleaned_data
from pipelines.data.steps.validate_features import validate_features
from pipelines.data.steps.validate_raw import validate_raw_data

docker = DockerSettings(
    parent_image="python:3.12-slim",
    python_package_installer="uv",
    python_package_installer_args={"system": None},
    requirements=[
        "zenml==0.96.4",
        "kagglehub>=0.4,<0.5",
        "pandas>=2.0,<2.4",
        "pandera>=0.20,<0.27",
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

orchestrator_settings = LocalDockerOrchestratorSettings(
    run_args={
        "cpu_count": 20,
        "mem_limit": "8g",
        "shm_size": "1g",
        "environment": {
            "KAGGLE_API_TOKEN": os.getenv("KAGGLE_API_TOKEN", ""),
        },
    }
)


@pipeline(
    settings={
        "docker": docker,
        "orchestrator.local_docker": orchestrator_settings,
    }
)
def data_pipeline() -> None:
    raw_tables, ingestion_metadata = ingest_data()
    validated_raw, _raw_validation_metadata = validate_raw_data(raw_tables, ingestion_metadata)
    profile_data(validated_raw)
    cleaned_tables, _cleaning_metadata = clean_data(validated_raw)
    validated_cleaned, _cleaned_validation_metadata = validate_cleaned_data(cleaned_tables)
    integrated_data, _join_metadata = integrate_data(validated_cleaned)
    transformed_data, _transform_metadata = transform_data(integrated_data)
    features, feature_metadata = engineer_features(transformed_data)
    validate_features(features, feature_metadata)


def main() -> None:
    data_pipeline()


if __name__ == "__main__":
    main()
