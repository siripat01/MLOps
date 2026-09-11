from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor

from mlops_project.features.covariates import to_timeseries_frame

DEFAULT_EVAL_METRICS = ("WQL", "RMSE", "RMSLE")


def evaluate_store_sales_predictor(
    model_path: Path,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = DEFAULT_EVAL_METRICS,
) -> dict[str, Any]:
    predictor = TimeSeriesPredictor.load(str(model_path))
    validation_ts = to_timeseries_frame(
        pd.concat([train_data, validation_data], ignore_index=True)
    )

    scores = predictor.evaluate(
        validation_ts,
        metrics=list(metrics),
        display=True,
    )

    normalized_scores = {str(key): float(value) for key, value in scores.items()}
    leaderboard = predictor.leaderboard(validation_ts, silent=True)
    return {
        "metrics": normalized_scores,
        "leaderboard": leaderboard.to_dict(orient="records"),
        "validation_rows": len(validation_data),
        "validation_items": int(validation_data["item_id"].nunique()),
    }
