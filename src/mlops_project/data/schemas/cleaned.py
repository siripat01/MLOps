from __future__ import annotations

import pandera.polars as pa
import polars as pl

from mlops_project.data.quality import validate_series_calendar

CLEANED_SCHEMAS = {
    "train": pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64, nullable=False, unique=True),
            "date": pa.Column(pl.Date, nullable=False),
            "store_nbr": pa.Column(pl.Int64, pa.Check.gt(0), nullable=False),
            "family": pa.Column(pl.String, nullable=False),
            "sales": pa.Column(pl.Float64, pa.Check.ge(0), nullable=False),
            "onpromotion": pa.Column(pl.Int64, pa.Check.ge(0), nullable=False),
        }
    ),
    "stores": pa.DataFrameSchema(
        {
            "store_nbr": pa.Column(pl.Int64, pa.Check.gt(0), nullable=False, unique=True),
            "city": pa.Column(pl.String, nullable=False),
            "state": pa.Column(pl.String, nullable=False),
            "type": pa.Column(pl.String, nullable=False),
            "cluster": pa.Column(pl.Int64, pa.Check.gt(0), nullable=False),
        }
    ),
    "oil": pa.DataFrameSchema(
        {
            "date": pa.Column(pl.Date, nullable=False, unique=True),
            "dcoilwtico": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True),
        }
    ),
    "holidays_events": pa.DataFrameSchema(
        {
            "date": pa.Column(pl.Date, nullable=False),
            "type": pa.Column(pl.String, nullable=False),
            "locale": pa.Column(
                pl.String,
                pa.Check.isin(["NATIONAL", "REGIONAL", "LOCAL"]),
                nullable=False,
            ),
            "locale_name": pa.Column(pl.String, nullable=False),
            "description": pa.Column(pl.String, nullable=False),
            "transferred": pa.Column(pl.Boolean, nullable=False),
        }
    ),
    "transactions": pa.DataFrameSchema(
        {
            "date": pa.Column(pl.Date, nullable=False),
            "store_nbr": pa.Column(pl.Int64, pa.Check.gt(0), nullable=False),
            "transactions": pa.Column(pl.Int64, pa.Check.ge(0), nullable=False),
        }
    ),
}


def validate_cleaned_tables(tables: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    missing = sorted(set(CLEANED_SCHEMAS) - set(tables))
    if missing:
        raise ValueError(f"Missing required cleaned tables: {missing}")

    validated = {
        name: schema.validate(tables[name]) for name, schema in CLEANED_SCHEMAS.items()
    }

    duplicated_transactions = validated["transactions"].select(
        pl.struct(["date", "store_nbr"]).is_duplicated().sum()
    ).item()
    if duplicated_transactions:
        raise ValueError(
            "transactions must be unique on ['date', 'store_nbr']; "
            f"found {duplicated_transactions} duplicates"
        )
    validate_series_calendar(
        validated["train"],
        table_name="Cleaned train",
        min_history_per_series=1,
        max_missing_calendar_dates=0,
    )
    return validated
