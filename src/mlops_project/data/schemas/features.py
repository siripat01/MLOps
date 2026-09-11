from __future__ import annotations

import pandera.polars as pa
import polars as pl

from mlops_project.data.features import TRAINING_COLUMNS
from mlops_project.data.quality import (
    missingness_report,
    validate_series_calendar,
)

_REQUIRED_FLOAT_HISTORY = {
    **{f"sales_lag_{window}": pa.Column(pl.Float64, nullable=True) for window in (1, 7, 14, 28)},
    **{
        f"sales_rolling_mean_{window}": pa.Column(pl.Float64, nullable=True)
        for window in (7, 14, 28)
    },
    **{
        f"sales_rolling_std_{window}": pa.Column(pl.Float64, nullable=True)
        for window in (7, 14, 28)
    },
    **{
        f"transactions_rolling_mean_{window}": pa.Column(pl.Float64, nullable=True)
        for window in (7, 14, 28)
    },
    **{
        f"transactions_rolling_std_{window}": pa.Column(pl.Float64, nullable=True)
        for window in (7, 14, 28)
    },
    "oil_lag_1": pa.Column(pl.Float64, nullable=True),
    "oil_lag_7": pa.Column(pl.Float64, nullable=True),
    "oil_rolling_mean_7": pa.Column(pl.Float64, nullable=True),
    "store_sales_lag_1": pa.Column(pl.Float64, nullable=True),
    "store_sales_rolling_mean_7": pa.Column(pl.Float64, nullable=True),
    "family_sales_lag_1": pa.Column(pl.Float64, nullable=True),
    "family_sales_rolling_mean_7": pa.Column(pl.Float64, nullable=True),
}

_REQUIRED_INT_HISTORY = {
    f"transactions_lag_{window}": pa.Column(pl.Int64, nullable=True)
    for window in (1, 7, 14, 28)
}

FEATURE_SCHEMA = pa.DataFrameSchema(
    {
        "id": pa.Column(pl.Int64, nullable=False, unique=True),
        "item_id": pa.Column(pl.String, nullable=False),
        "date": pa.Column(pl.Date, nullable=False),
        "store_nbr": pa.Column(pl.Int64, pa.Check.gt(0), nullable=False),
        "family": pa.Column(pl.String, nullable=False),
        "sales": pa.Column(pl.Float64, pa.Check.ge(0), nullable=False),
        "onpromotion": pa.Column(pl.Int64, pa.Check.ge(0), nullable=False),
        "city": pa.Column(pl.String, nullable=False),
        "state": pa.Column(pl.String, nullable=False),
        "store_type": pa.Column(pl.String, nullable=False),
        "cluster": pa.Column(pl.Int64, pa.Check.gt(0), nullable=False),
        "is_holiday": pa.Column(pl.Boolean, nullable=False),
        "holiday_type": pa.Column(pl.String, nullable=False),
        "holiday_descriptions": pa.Column(pl.String, nullable=False),
        "holiday_count": pa.Column(pl.Int64, pa.Check.ge(0), nullable=False),
        "year": pa.Column(pl.Int32, nullable=False),
        "month": pa.Column(pl.Int8, nullable=False),
        "quarter": pa.Column(pl.Int8, nullable=False),
        "week_of_year": pa.Column(pl.Int8, nullable=False),
        "day_of_week": pa.Column(pl.Int8, nullable=False),
        "day_of_month": pa.Column(pl.Int8, nullable=False),
        "is_weekend": pa.Column(pl.Boolean, nullable=False),
        "is_payday": pa.Column(pl.Boolean, nullable=False),
        "is_month_end": pa.Column(pl.Boolean, nullable=False),
        "is_quarter_end": pa.Column(pl.Boolean, nullable=False),
        "is_year_end": pa.Column(pl.Boolean, nullable=False),
        "is_onpromotion": pa.Column(pl.Boolean, nullable=False),
        "promotion_log1p": pa.Column(pl.Float64, pa.Check.ge(0), nullable=False),
        "promotion_capped_10": pa.Column(pl.Int64, pa.Check.ge(0), nullable=False),
        "holiday_type_BRIDGE": pa.Column(pl.Boolean, nullable=False),
        "holiday_type_EVENT": pa.Column(pl.Boolean, nullable=False),
        "holiday_type_HOLIDAY": pa.Column(pl.Boolean, nullable=False),
        "holiday_type_TRANSFER": pa.Column(pl.Boolean, nullable=False),
        "holiday_type_WORK_DAY": pa.Column(pl.Boolean, nullable=False),
        **_REQUIRED_FLOAT_HISTORY,
        **_REQUIRED_INT_HISTORY,
    },
    strict=False,
)

REQUIRED_FEATURE_COLUMNS = set(FEATURE_SCHEMA.columns)
FORBIDDEN_CURRENT_OBSERVED_COLUMNS = {"transactions", "dcoilwtico"}


def validate_feature_table(
    df: pl.DataFrame,
    *,
    min_history_per_series: int = 1,
    max_missing_calendar_dates: int = 0,
    max_training_missing_rate: float = 0.0,
) -> pl.DataFrame:
    missing = sorted(REQUIRED_FEATURE_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    missing_training_columns = sorted(set(TRAINING_COLUMNS) - set(df.columns))
    if missing_training_columns:
        raise ValueError(
            f"Missing training-consumed feature columns: {missing_training_columns}"
        )

    forbidden = sorted(FORBIDDEN_CURRENT_OBSERVED_COLUMNS & set(df.columns))
    if forbidden:
        raise ValueError(
            "Current observed signals must not be model-facing features; "
            f"found {forbidden}"
        )

    validated = FEATURE_SCHEMA.validate(df)
    validate_series_calendar(
        validated,
        table_name="Feature table",
        min_history_per_series=min_history_per_series,
        max_missing_calendar_dates=max_missing_calendar_dates,
    )

    training_missingness = missingness_report(validated, TRAINING_COLUMNS)
    excessive_training_missingness = {
        column: rate
        for column, rate in training_missingness.items()
        if rate > max_training_missing_rate
    }
    if excessive_training_missingness:
        raise ValueError(
            "Training-consumed feature columns exceed missingness threshold: "
            f"{excessive_training_missingness}"
        )

    comparable = validated.filter(pl.col("sales_lag_1").is_not_null())
    if comparable.height and comparable.select(
        (pl.col("sales") == pl.col("sales_lag_1")).all()
    ).item():
        raise ValueError("Suspicious feature: sales_lag_1 is identical to sales for every row")

    return validated
