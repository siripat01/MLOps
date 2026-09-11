from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from zenml import log_metadata, step

from mlops_project.models.eval import DEFAULT_EVAL_METRICS, evaluate_store_sales_predictor

EXPERIMENT_TRACKER_NAME = os.getenv("ZENML_EXPERIMENT_TRACKER_NAME") or os.getenv(
    "MLFLOW_EXPERIMENT_TRACKER_NAME"
)


def _log_mlflow_evaluation(report: dict[str, Any]) -> None:
    if mlflow.active_run() is None:
        return
    for metric_name, metric_value in report["metrics"].items():
        mlflow.log_metric(f"validation_{metric_name}", metric_value)
    mlflow.log_metric("validation_rows", report["validation_rows"])
    mlflow.log_metric("validation_items", report["validation_items"])


@step(enable_cache=False, experiment_tracker=EXPERIMENT_TRACKER_NAME)
def evaluate_model(
    model_path: Path,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    training_metadata: dict[str, Any],
    metrics: tuple[str, ...] = DEFAULT_EVAL_METRICS,
) -> dict[str, float]:
    report = evaluate_store_sales_predictor(
        model_path,
        train_data,
        validation_data,
        metrics=metrics,
    )
    metadata = {
        "evaluation": {
            "metrics": report["metrics"],
            "validation_rows": report["validation_rows"],
            "validation_items": report["validation_items"],
            "trained_model_path": training_metadata.get("model_path"),
            "best_model": training_metadata.get("best_model"),
        }
    }
    _log_mlflow_evaluation(report)
    log_metadata(metadata=metadata)
    return report["metrics"]
