from __future__ import annotations

from pathlib import Path

import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from mlops_project.serving.schemas import (
    ForecastPoint,
    ForecastRequest,
    ForecastResponse,
    HistoryPoint,
    KnownCovariatePoint,
)


class StoreSalesForecaster:
    """Adapter between the HTTP contract and AutoGluon TimeSeriesPredictor."""

    def __init__(self, predictor: TimeSeriesPredictor) -> None:
        self._predictor = predictor

    @classmethod
    def load(cls, model_path: Path) -> StoreSalesForecaster:
        predictor = TimeSeriesPredictor.load(str(model_path))
        return cls(predictor)

    def predict(self, request: ForecastRequest) -> ForecastResponse:
        history = self._history_frame(request.history)
        known_covariates = self._known_covariates_frame(request.known_covariates)

        predictions = self._predictor.predict(
            history,
            known_covariates=known_covariates,
        )
        return ForecastResponse(predictions=self._serialize(predictions))

    @staticmethod
    def _history_frame(points: list[HistoryPoint]) -> TimeSeriesDataFrame:
        frame = pd.DataFrame(point.model_dump() for point in points)
        timeseries = TimeSeriesDataFrame.from_data_frame(
            frame,
            id_column="item_id",
            timestamp_column="date",
        ).convert_frequency(freq="D")

        timeseries["onpromotion"] = timeseries["onpromotion"].fillna(0)
        timeseries["is_holiday"] = timeseries["is_holiday"].fillna(False)
        return timeseries

    @staticmethod
    def _known_covariates_frame(
        points: list[KnownCovariatePoint],
    ) -> TimeSeriesDataFrame:
        frame = pd.DataFrame(point.model_dump() for point in points)
        timeseries = TimeSeriesDataFrame.from_data_frame(
            frame,
            id_column="item_id",
            timestamp_column="date",
        ).convert_frequency(freq="D")

        timeseries["onpromotion"] = timeseries["onpromotion"].fillna(0)
        timeseries["is_holiday"] = timeseries["is_holiday"].fillna(False)
        return timeseries

    @staticmethod
    def _serialize(predictions: TimeSeriesDataFrame) -> list[ForecastPoint]:
        rows = predictions.reset_index().to_dict(orient="records")
        serialized: list[ForecastPoint] = []

        for row in rows:
            item_id = str(row.pop("item_id"))
            timestamp = pd.Timestamp(row.pop("timestamp")).to_pydatetime()
            values = {str(key): float(value) for key, value in row.items()}
            serialized.append(
                ForecastPoint(
                    item_id=item_id,
                    timestamp=timestamp,
                    values=values,
                )
            )

        return serialized
