from __future__ import annotations

import polars as pl
import pandera.polars as pa


# Raw validation is intentionally structural. Value-level rules belong after cleaning
# so malformed-but-repairable rows can reach the cleaning stage.
RAW_SCHEMAS = {
    "train": pa.DataFrameSchema(
        {
            "id": pa.Column(pl.Int64, nullable=True),
            "date": pa.Column(pl.Date, nullable=True),
            "store_nbr": pa.Column(pl.Int64, nullable=True),
            "family": pa.Column(pl.String, nullable=True),
            "sales": pa.Column(pl.Float64, nullable=True),
            "onpromotion": pa.Column(pl.Int64, nullable=True),
        }
    ),
    "stores": pa.DataFrameSchema(
        {
            "store_nbr": pa.Column(pl.Int64, nullable=True),
            "city": pa.Column(pl.String, nullable=True),
            "state": pa.Column(pl.String, nullable=True),
            "type": pa.Column(pl.String, nullable=True),
            "cluster": pa.Column(pl.Int64, nullable=True),
        }
    ),
    "oil": pa.DataFrameSchema(
        {
            "date": pa.Column(pl.Date, nullable=True),
            "dcoilwtico": pa.Column(pl.Float64, nullable=True),
        }
    ),
    "holidays_events": pa.DataFrameSchema(
        {
            "date": pa.Column(pl.Date, nullable=True),
            "type": pa.Column(pl.String, nullable=True),
            "locale": pa.Column(pl.String, nullable=True),
            "locale_name": pa.Column(pl.String, nullable=True),
            "description": pa.Column(pl.String, nullable=True),
            "transferred": pa.Column(pl.Boolean, nullable=True),
        }
    ),
    "transactions": pa.DataFrameSchema(
        {
            "date": pa.Column(pl.Date, nullable=True),
            "store_nbr": pa.Column(pl.Int64, nullable=True),
            "transactions": pa.Column(pl.Int64, nullable=True),
        }
    ),
}


def validate_raw_tables(tables: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    missing = sorted(set(RAW_SCHEMAS) - set(tables))
    if missing:
        raise ValueError(f"Missing required raw tables: {missing}")
    return {name: schema.validate(tables[name]) for name, schema in RAW_SCHEMAS.items()}
