import os
from pathlib import Path
from typing import Annotated, Any, Tuple  # noqa: UP035

import mlflow
import pandas as pd
from zenml import ArtifactConfig, log_metadata, step
from zenml.enums import ArtifactType

from mlops_project.models.training import train_store_sales_predictor

MODEL_ARTIFACT_NAME = os.getenv("MODEL_ARTIFACT_NAME", "store_sales_model")
EXPERIMENT_TRACKER_NAME = os.getenv("ZENML_EXPERIMENT_TRACKER_NAME") or os.getenv(
    "MLFLOW_EXPERIMENT_TRACKER_NAME"
)


def _log_mlflow_training(metadata: dict[str, Any]) -> None:
    if mlflow.active_run() is None:
        return
    mlflow.log_params(
        {
            "prediction_length": metadata["prediction_length"],
            "presets": metadata["presets"],
            "eval_metric": metadata["eval_metric"],
            "time_limit": metadata["time_limit"],
            "enable_ensemble": metadata["enable_ensemble"],
            "known_covariate_count": len(metadata["known_covariates"]),
        }
    )
    mlflow.log_metric("train_rows", metadata["train_rows"])
    mlflow.log_metric("train_items", metadata["train_items"])
    mlflow.log_metric("model_count", metadata["model_count"])


@step(enable_cache=False, experiment_tracker=EXPERIMENT_TRACKER_NAME)
def train_model(
    train_data: pd.DataFrame,
    prediction_length: int = 16,
    presets: str = "high_quality",
    eval_metric: str = "RMSLE",
    time_limit: int | None = None,
    enable_ensemble: bool = True,
    model_profile: str | None = "local_safe",
) -> Tuple[  # noqa: UP006
    Annotated[
        Path,
        ArtifactConfig(
            name=MODEL_ARTIFACT_NAME,
            artifact_type=ArtifactType.MODEL,
            tags=["autogluon", "store-sales"],
        ),
    ],
    Annotated[
        dict[str, Any],
        ArtifactConfig(
            name=f"{MODEL_ARTIFACT_NAME}_training_metadata",
            tags=["autogluon", "training-metadata", "store-sales"],
        ),
    ],
]:
    model_path, metadata = train_store_sales_predictor(
        train_data,
        prediction_length=prediction_length,
        presets=presets,
        eval_metric=eval_metric,
        time_limit=time_limit,
        enable_ensemble=enable_ensemble,
        model_profile=model_profile,
    )
    _log_mlflow_training(metadata)
    log_metadata(
        metadata={
            "training": {key: value for key, value in metadata.items() if key != "leaderboard"}
        }
    )
    return model_path, metadata
