from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor

from app.schemas import ModelMetadata, PredictionPoint, PredictionRequest, PredictionResponse

HOLIDAY_TYPES = ("BRIDGE", "EVENT", "HOLIDAY", "TRANSFER", "WORK DAY")
KNOWN_COVARIATES = (
    "onpromotion",
    "is_onpromotion",
    "promotion_log1p",
    "promotion_capped_10",
    "is_holiday",
    "holiday_count",
    "is_payday",
    "is_month_end",
    "is_quarter_end",
    "is_year_end",
    "year",
    "month",
    "quarter",
    "week_of_year",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "holiday_type_BRIDGE",
    "holiday_type_EVENT",
    "holiday_type_HOLIDAY",
    "holiday_type_TRANSFER",
    "holiday_type_WORK_DAY",
)


def _add_known_covariates(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    timestamps = pd.to_datetime(enriched["date"])
    if "onpromotion" not in enriched:
        enriched["onpromotion"] = 0
    if "is_holiday" not in enriched:
        enriched["is_holiday"] = False
    if "holiday_count" not in enriched:
        enriched["holiday_count"] = enriched["is_holiday"].astype(int)
    if "holiday_type" not in enriched:
        enriched["holiday_type"] = "NONE"
    enriched["onpromotion"] = pd.to_numeric(enriched["onpromotion"], errors="coerce").fillna(0)
    enriched["is_holiday"] = enriched["is_holiday"].fillna(False).astype(bool)
    enriched["holiday_count"] = (
        pd.to_numeric(enriched["holiday_count"], errors="coerce")
        .fillna(0)
        .astype("int64")
    )
    holiday_type = enriched["holiday_type"].fillna("NONE").astype(str).str.upper()
    enriched["is_onpromotion"] = enriched["onpromotion"] > 0
    enriched["promotion_log1p"] = np.log1p(enriched["onpromotion"].clip(lower=0))
    enriched["promotion_capped_10"] = (
        enriched["onpromotion"].clip(lower=0, upper=10).astype("int64")
    )
    enriched["year"] = timestamps.dt.year
    enriched["month"] = timestamps.dt.month
    enriched["quarter"] = timestamps.dt.quarter
    enriched["week_of_year"] = timestamps.dt.isocalendar().week.astype("int64")
    enriched["day_of_week"] = timestamps.dt.weekday + 1
    enriched["day_of_month"] = timestamps.dt.day
    enriched["is_weekend"] = enriched["day_of_week"] >= 6
    enriched["is_payday"] = enriched["day_of_month"] == 15
    enriched["is_month_end"] = timestamps.dt.is_month_end
    enriched["is_quarter_end"] = timestamps.dt.is_quarter_end
    enriched["is_year_end"] = timestamps.dt.is_year_end
    for holiday in HOLIDAY_TYPES:
        enriched[f"holiday_type_{holiday.replace(' ', '_')}"] = holiday_type == holiday
    for column in KNOWN_COVARIATES:
        if column not in enriched:
            enriched[column] = False if column.startswith(("is_", "holiday_type_")) else 0
    return enriched


class AutoGluonModel:
    def __init__(self, predictor: TimeSeriesPredictor, metadata: ModelMetadata) -> None:
        self.predictor = predictor
        self._metadata = metadata

    @classmethod
    def load(cls, model_path: str | Any, metadata: ModelMetadata) -> AutoGluonModel:
        return cls(TimeSeriesPredictor.load(str(model_path)), metadata)

    def metadata(self) -> ModelMetadata:
        return self._metadata

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        validate_prediction_request(request, self._metadata.prediction_length)
        history = _to_timeseries(request.history, include_sales=True)
        known = _to_timeseries(request.known_covariates, include_sales=False)
        predictions = self.predictor.predict(history, known_covariates=known)
        rows = predictions.reset_index().to_dict(orient="records")
        return PredictionResponse(
            predictions=[
                PredictionPoint(
                    item_id=str(row.pop("item_id")),
                    timestamp=pd.Timestamp(row.pop("timestamp")).to_pydatetime(),
                    values={str(key): float(value) for key, value in row.items()},
                )
                for row in rows
            ]
        )


def validate_prediction_request(request: PredictionRequest, prediction_length: int) -> None:
    if prediction_length < 1:
        raise ValueError("model prediction_length must be positive")

    history = pd.DataFrame(point.model_dump() for point in request.history)
    known = pd.DataFrame(point.model_dump() for point in request.known_covariates)
    history["date"] = pd.to_datetime(history["date"])
    known["date"] = pd.to_datetime(known["date"])
    history_items = set(history["item_id"])
    known_items = set(known["item_id"])
    if known_items != history_items:
        raise ValueError(
            "known_covariates item_id values must exactly match history item_id values"
        )

    for item_id, item_history in history.groupby("item_id"):
        if item_history["date"].duplicated().any():
            raise ValueError(f"history contains duplicate dates for item_id={item_id}")
        item_known = known.loc[known["item_id"] == item_id]
        if item_known["date"].duplicated().any():
            raise ValueError(f"known_covariates contains duplicate dates for item_id={item_id}")
        last_date = item_history["date"].max()
        expected = pd.date_range(
            last_date + pd.Timedelta(1, unit="D"), periods=prediction_length, freq="D"
        )
        actual = pd.DatetimeIndex(item_known["date"].sort_values())
        if len(actual) != prediction_length or not actual.equals(expected):
            raise ValueError(
                f"known_covariates must contain the next {prediction_length} future daily dates "
                f"after history for item_id={item_id}"
            )


def _to_timeseries(points: list[Any], *, include_sales: bool) -> TimeSeriesDataFrame:
    frame = pd.DataFrame(point.model_dump() for point in points)
    if not include_sales:
        frame = frame.drop(columns=["sales"], errors="ignore")
    enriched = _add_known_covariates(frame)
    return TimeSeriesDataFrame.from_data_frame(
        enriched, id_column="item_id", timestamp_column="date"
    ).convert_frequency(freq="D")
