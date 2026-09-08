from __future__ import annotations

from typing import Any

import polars as pl


def assert_unique(df: pl.DataFrame, keys: list[str], table_name: str) -> None:
    duplicates = df.select(pl.struct(keys).is_duplicated().sum()).item()
    if duplicates:
        raise ValueError(f"{table_name} must be unique on {keys}; found {duplicates} duplicates")


def integrate_store_sales(tables: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, dict[str, Any]]:
    train = tables["train"]
    stores = tables["stores"].rename({"type": "store_type"})
    oil = tables["oil"]
    transactions = tables["transactions"]
    holidays = aggregate_holidays(tables["holidays_events"])

    assert_unique(stores, ["store_nbr"], "stores")
    assert_unique(oil, ["date"], "oil")
    assert_unique(transactions, ["date", "store_nbr"], "transactions")
    assert_unique(holidays, ["date"], "holidays")

    metrics: dict[str, Any] = {"rows_before": train.height}
    integrated = (
        train.join(stores, on="store_nbr", how="left")
        .join(oil, on="date", how="left")
        .join(transactions, on=["date", "store_nbr"], how="left")
        .join(holidays, on="date", how="left")
        .with_columns(
            pl.col("is_holiday").fill_null(False),
            pl.col("holiday_type").fill_null("NONE"),
        )
    )
    metrics["rows_after"] = integrated.height
    metrics["unmatched_stores"] = integrated.filter(pl.col("city").is_null()).height
    metrics["unmatched_oil_dates"] = integrated.filter(pl.col("dcoilwtico").is_null()).height
    if integrated.height != train.height:
        raise ValueError(
            f"Unexpected join row expansion: before={train.height}, after={integrated.height}"
        )
    return integrated, metrics


def aggregate_holidays(holidays: pl.DataFrame) -> pl.DataFrame:
    return (
        holidays.group_by("date")
        .agg(
            pl.lit(True).alias("is_holiday"),
            pl.col("type").sort().first().alias("holiday_type"),
            pl.col("description").str.join("; ").alias("holiday_descriptions"),
        )
        .sort("date")
    )
