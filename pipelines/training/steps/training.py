from __future__ import annotations

from pathlib import Path

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from zenml import step


@step(
    enable_cache=False,
)
def train_model(
    train_data: pd.DataFrame,
    prediction_length: int = 16,
) -> Path:
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
            "DeepAR did not produce a fitted model. Check the training logs "
            "for CUDA or data compatibility errors."
        )

    print(predictor.leaderboard())

    return model_path
