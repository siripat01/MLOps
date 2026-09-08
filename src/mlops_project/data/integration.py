from __future__ import annotations

from typing import Any

import polars as pl


def assert_unique(df: pl.DataFrame, keys: list[str], table_name: str) -> None:
    duplicates = df.select(pl.struct(keys).is_duplicated().sum()).item()
    if duplicates:
        raise ValueError(f"{table_name} must be unique on {keys}; found {duplicates} duplicates")


def scope_holidays_to_stores(
    holidays: pl.DataFrame,
    stores: pl.DataFrame,
) -> pl.DataFrame:
    """Resolve national/regional/local holidays to the stores they actually affect."""
    store_geo = stores.select(["store_nbr", "city", "state"])

    national = holidays.filter(pl.col("locale") == "NATIONAL").join(
        store_geo.select("store_nbr"),
        how="cross",
    )
    regional = holidays.filter(pl.col("locale") == "REGIONAL").join(
        store_geo.select(["store_nbr", "state"]),
        left_on="locale_name",
        right_on="state",
        how="inner",
    )
    local = holidays.filter(pl.col("locale") == "LOCAL").join(
        store_geo.select(["store_nbr", "city"]),
        left_on="locale_name",
        right_on="city",
        how="inner",
    )

    scoped = pl.concat([national, regional, local], how="diagonal_relaxed")
    if scoped.is_empty():
        return pl.DataFrame(
            schema={
                "date": pl.Date,
                "store_nbr": pl.Int64,
                "is_holiday": pl.Boolean,
                "holiday_type": pl.String,
                "holiday_descriptions": pl.String,
                "holiday_count": pl.Int64,
            }
        )

    return (
        scoped.group_by(["date", "store_nbr"])
        .agg(
            pl.lit(True).alias("is_holiday"),
            pl.col("type").sort().first().alias("holiday_type"),
            pl.col("description").str.join("; ").alias("holiday_descriptions"),
            pl.len().cast(pl.Int64).alias("holiday_count"),
        )
        .sort(["store_nbr", "date"])
    )


def integrate_store_sales(tables: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, dict[str, Any]]:
    train = tables["train"]
    stores = tables["stores"].rename({"type": "store_type"})
    oil = tables["oil"]
    transactions = tables["transactions"]
    holidays = scope_holidays_to_stores(tables["holidays_events"], stores)

    assert_unique(stores, ["store_nbr"], "stores")
    assert_unique(oil, ["date"], "oil")
    assert_unique(transactions, ["date", "store_nbr"], "transactions")
    assert_unique(holidays, ["date", "store_nbr"], "holidays")

    metrics: dict[str, Any] = {"rows_before": train.height}
    integrated = (
        train.join(stores, on="store_nbr", how="left")
        .join(oil, on="date", how="left")
        .join(transactions, on=["date", "store_nbr"], how="left")
        .join(holidays, on=["date", "store_nbr"], how="left")
        .with_columns(
            pl.col("is_holiday").fill_null(False),
            pl.col("holiday_type").fill_null("NONE"),
            pl.col("holiday_descriptions").fill_null("NONE"),
            pl.col("holiday_count").fill_null(0).cast(pl.Int64),
        )
    )

    metrics["rows_after"] = integrated.height
    metrics["unmatched_stores"] = integrated.filter(pl.col("city").is_null()).height
    metrics["unmatched_oil_dates"] = integrated.filter(pl.col("dcoilwtico").is_null()).height
    metrics["unmatched_transactions"] = integrated.filter(pl.col("transactions").is_null()).height
    metrics["holiday_rows"] = integrated.filter(pl.col("is_holiday")).height

    if integrated.height != train.height:
        raise ValueError(
            f"Unexpected join row expansion: before={train.height}, after={integrated.height}"
        )
    return integrated, metrics
