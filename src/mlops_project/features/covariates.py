from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame

from mlops_project.data.features import KNOWN_COVARIATE_COLUMNS, TRAINING_COLUMNS

HOLIDAY_TYPES = ("BRIDGE", "EVENT", "HOLIDAY", "TRANSFER", "WORK DAY")


def _date_series(frame: pd.DataFrame) -> pd.Series:
    if "date" in frame.columns:
        return pd.to_datetime(frame["date"])
    if "timestamp" in frame.columns:
        return pd.to_datetime(frame["timestamp"])
    raise ValueError("Expected a date or timestamp column")


def add_known_covariates(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    timestamps = _date_series(enriched)

    if "onpromotion" not in enriched.columns:
        enriched["onpromotion"] = 0
    if "is_holiday" not in enriched.columns:
        enriched["is_holiday"] = False
    if "holiday_count" not in enriched.columns:
        enriched["holiday_count"] = enriched["is_holiday"].astype(int)
    if "holiday_type" not in enriched.columns:
        enriched["holiday_type"] = "HOLIDAY"
        enriched.loc[~enriched["is_holiday"].astype(bool), "holiday_type"] = "NONE"

    enriched["onpromotion"] = pd.to_numeric(enriched["onpromotion"], errors="coerce").fillna(0)
    enriched["is_holiday"] = enriched["is_holiday"].fillna(False).astype(bool)
    enriched["holiday_count"] = (
        pd.to_numeric(enriched["holiday_count"], errors="coerce").fillna(0).astype("int64")
    )

    enriched["is_onpromotion"] = enriched["onpromotion"] > 0
    enriched["promotion_log1p"] = np.log1p(enriched["onpromotion"].clip(lower=0))
    enriched["promotion_capped_10"] = enriched["onpromotion"].clip(lower=0, upper=10).astype(
        "int64"
    )

    enriched["year"] = timestamps.dt.year.astype("int64")
    enriched["month"] = timestamps.dt.month.astype("int64")
    enriched["quarter"] = timestamps.dt.quarter.astype("int64")
    enriched["week_of_year"] = timestamps.dt.isocalendar().week.astype("int64")
    enriched["day_of_week"] = (timestamps.dt.weekday + 1).astype("int64")
    enriched["day_of_month"] = timestamps.dt.day.astype("int64")
    enriched["is_weekend"] = enriched["day_of_week"] >= 6
    enriched["is_payday"] = enriched["day_of_month"] == 15
    enriched["is_month_end"] = timestamps.dt.is_month_end
    enriched["is_quarter_end"] = timestamps.dt.is_quarter_end
    enriched["is_year_end"] = timestamps.dt.is_year_end

    holiday_type = enriched["holiday_type"].fillna("NONE").astype(str).str.upper()
    for holiday in HOLIDAY_TYPES:
        enriched[f"holiday_type_{holiday.replace(' ', '_')}"] = holiday_type == holiday

    for column in KNOWN_COVARIATE_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = False if column.startswith(("is_", "holiday_type_")) else 0

    return enriched


def training_frame_from_features(df: Any) -> pd.DataFrame:
    frame = df.to_pandas() if hasattr(df, "to_pandas") else pd.DataFrame(df)
    missing = sorted({"item_id", "date", "sales"} - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required training columns: {missing}")
    return add_known_covariates(frame).loc[:, TRAINING_COLUMNS]


def to_timeseries_frame(frame: pd.DataFrame) -> TimeSeriesDataFrame:
    timeseries = TimeSeriesDataFrame.from_data_frame(
        add_known_covariates(frame),
        id_column="item_id",
        timestamp_column="date",
    ).convert_frequency(freq="D")
    expanded = timeseries.reset_index().rename(columns={"timestamp": "date"})
    return TimeSeriesDataFrame.from_data_frame(
        add_known_covariates(expanded),
        id_column="item_id",
        timestamp_column="date",
    )


def records_to_known_covariates(records: list[dict[str, Any]]) -> TimeSeriesDataFrame:
    return to_timeseries_frame(pd.DataFrame(records).drop(columns=["sales"], errors="ignore"))


def default_known_covariate_record(
    *,
    item_id: str,
    forecast_date: date,
    onpromotion: float = 0,
    is_holiday: bool = False,
    holiday_count: int | None = None,
    holiday_type: str | None = None,
) -> dict[str, Any]:
    return {
        "item_id": item_id,
        "date": forecast_date,
        "onpromotion": onpromotion,
        "is_holiday": is_holiday,
        "holiday_count": holiday_count if holiday_count is not None else int(is_holiday),
        "holiday_type": holiday_type or ("HOLIDAY" if is_holiday else "NONE"),
    }
