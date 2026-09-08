from __future__ import annotations

import pandera.polars as pa
import polars as pl

TRAIN_SCHEMA = pa.DataFrameSchema(
    {
        "id": pa.Column(pl.Int64, nullable=False, unique=True),
        "date": pa.Column(pl.Date, nullable=False),
        "store_nbr": pa.Column(pl.Int64, pa.Check.gt(0), nullable=False),
        "family": pa.Column(pl.String, nullable=False),
        "sales": pa.Column(pl.Float64, pa.Check.ge(0), nullable=False),
        "onpromotion": pa.Column(pl.Int64, pa.Check.ge(0), nullable=False),
    }
)

STORES_SCHEMA = pa.DataFrameSchema(
    {
        "store_nbr": pa.Column(pl.Int64, pa.Check.gt(0), nullable=False, unique=True),
        "city": pa.Column(pl.String, nullable=False),
        "state": pa.Column(pl.String, nullable=False),
        "type": pa.Column(pl.String, nullable=False),
        "cluster": pa.Column(pl.Int64, pa.Check.gt(0), nullable=False),
    }
)

OIL_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column(pl.Date, nullable=False),
        "dcoilwtico": pa.Column(pl.Float64, pa.Check.ge(0), nullable=True),
    }
)

HOLIDAYS_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column(pl.Date, nullable=False),
        "type": pa.Column(pl.String, nullable=False),
        "locale": pa.Column(pl.String, nullable=False),
        "locale_name": pa.Column(pl.String, nullable=False),
        "description": pa.Column(pl.String, nullable=False),
        "transferred": pa.Column(pl.Boolean, nullable=False),
    }
)

TRANSACTIONS_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column(pl.Date, nullable=False),
        "store_nbr": pa.Column(pl.Int64, pa.Check.gt(0), nullable=False),
        "transactions": pa.Column(pl.Int64, pa.Check.ge(0), nullable=False),
    }
)

RAW_SCHEMAS = {
    "train": TRAIN_SCHEMA,
    "stores": STORES_SCHEMA,
    "oil": OIL_SCHEMA,
    "holidays_events": HOLIDAYS_SCHEMA,
    "transactions": TRANSACTIONS_SCHEMA,
}


def validate_raw_tables(tables: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    missing = sorted(set(RAW_SCHEMAS) - set(tables))
    if missing:
        raise ValueError(f"Missing required raw tables: {missing}")
    return {name: schema.validate(tables[name]) for name, schema in RAW_SCHEMAS.items()}
