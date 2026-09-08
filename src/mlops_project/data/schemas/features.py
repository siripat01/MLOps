from __future__ import annotations

import pandera.polars as pa
import polars as pl

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
        **_REQUIRED_FLOAT_HISTORY,
        **_REQUIRED_INT_HISTORY,
    },
    strict=False,
)

REQUIRED_FEATURE_COLUMNS = set(FEATURE_SCHEMA.columns)
FORBIDDEN_CURRENT_OBSERVED_COLUMNS = {"transactions", "dcoilwtico"}


def validate_feature_table(df: pl.DataFrame) -> pl.DataFrame:
    missing = sorted(REQUIRED_FEATURE_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    forbidden = sorted(FORBIDDEN_CURRENT_OBSERVED_COLUMNS & set(df.columns))
    if forbidden:
        raise ValueError(
            "Current observed signals must not be model-facing features; "
            f"found {forbidden}"
        )

    validated = FEATURE_SCHEMA.validate(df)
    duplicated_keys = validated.select(
        pl.struct(["store_nbr", "family", "date"]).is_duplicated().sum()
    ).item()
    if duplicated_keys:
        raise ValueError(f"Feature table has duplicate store/family/date keys: {duplicated_keys}")

    comparable = validated.filter(pl.col("sales_lag_1").is_not_null())
    if comparable.height and comparable.select(
        (pl.col("sales") == pl.col("sales_lag_1")).all()
    ).item():
        raise ValueError("Suspicious feature: sales_lag_1 is identical to sales for every row")

    return validated
