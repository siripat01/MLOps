from __future__ import annotations

import pandera.polars as pa
import polars as pl

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
        "dcoilwtico": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True),
        "is_holiday": pa.Column(pl.Boolean, nullable=False),
        "transactions": pa.Column(pl.Int64, pa.Check.ge(0), nullable=True),
    },
    strict=False,
)


REQUIRED_FEATURE_COLUMNS = set(FEATURE_SCHEMA.columns)


def validate_feature_table(df: pl.DataFrame) -> pl.DataFrame:
    missing = sorted(REQUIRED_FEATURE_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    validated = FEATURE_SCHEMA.validate(df)
    duplicated_keys = validated.select(
        pl.struct(["store_nbr", "family", "date"]).is_duplicated().sum()
    ).item()
    if duplicated_keys:
        raise ValueError(f"Feature table has duplicate store/family/date keys: {duplicated_keys}")
    if validated.select((pl.col("sales") == pl.col("lag_1")).all()).item():
        raise ValueError("Suspicious feature: lag_1 is identical to sales for every row")
    return validated
