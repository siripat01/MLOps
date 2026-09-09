from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from zenml import ArtifactConfig, step
from zenml.enums import ArtifactType

MODEL_ARTIFACT_NAME = os.getenv("MODEL_ARTIFACT_NAME", "store_sales_model")


@step(enable_cache=False)
def train_model(
    train_data: pd.DataFrame,
    prediction_length: int = 16,
) -> Annotated[
    Path,
    ArtifactConfig(
        name=MODEL_ARTIFACT_NAME,
        artifact_type=ArtifactType.MODEL,
        tags=["autogluon", "store-sales"],
    ),
]:
    train_ts = TimeSeriesDataFrame.from_data_frame(
        train_data,
        id_column="item_id",
        timestamp_column="date",
    )

    # Store Sales is a daily forecasting problem.
    train_ts = train_ts.convert_frequency(freq="D")

    # Fill known covariates generated for missing dates.
    train_ts["onpromotion"] = train_ts["onpromotion"].fillna(0)
    train_ts["is_holiday"] = train_ts["is_holiday"].fillna(False)

    model_path = Path("artifacts/autogluon/store_sales").resolve()

    predictor = TimeSeriesPredictor(
        target="sales",
        prediction_length=prediction_length,
        freq="D",
        known_covariates_names=[
            "onpromotion",
            "is_holiday",
        ],
        eval_metric="RMSLE",
        path=str(model_path),
    )

    predictor.fit(
        train_ts,
        presets="best_quality",
        enable_ensemble=True,
        verbosity=3,
    )

    fitted_models = predictor.model_names()
    if not fitted_models:
        raise RuntimeError(
            "Training did not produce a fitted model. Check the logs for CUDA "
            "or data compatibility errors."
        )

    print(predictor.leaderboard())
    return model_path
