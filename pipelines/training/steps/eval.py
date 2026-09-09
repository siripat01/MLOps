from pathlib import Path

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from zenml import step


@step(enable_cache=False)
def evaluate_model(
    model_path: Path,
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
) -> dict[str, float]:
    predictor = TimeSeriesPredictor.load(str(model_path))

    validation_ts = TimeSeriesDataFrame.from_data_frame(
        pd.concat([train_data, validation_data], ignore_index=True),
        id_column="item_id",
        timestamp_column="date",
    )

    return predictor.evaluate(
        validation_ts,
        metrics=["WQL", "RMSE", "RMSLE"],
        display=True,
    )
