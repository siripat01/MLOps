from __future__ import annotations

from typing import Any

import polars as pl


def normalize_text_columns(df: pl.DataFrame) -> pl.DataFrame:
    expressions = [
        pl.col(col).str.strip_chars().str.to_uppercase().alias(col)
        for col, dtype in df.schema.items()
        if dtype == pl.String
    ]
    return df.with_columns(expressions) if expressions else df


def clean_train(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    before = df.height
    cleaned = (
        normalize_text_columns(df)
        .unique(subset=["id"], keep="first")
        .filter(
            pl.col("date").is_not_null()
            & pl.col("store_nbr").is_not_null()
            & pl.col("family").is_not_null()
            & pl.col("sales").is_not_null()
            & (pl.col("sales") >= 0)
            & (pl.col("onpromotion") >= 0)
        )
    )
    return cleaned, {"rows_before": before, "rows_after": cleaned.height}


def clean_stores(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    before = df.height
    cleaned = normalize_text_columns(df).unique(subset=["store_nbr"], keep="first")
    return cleaned, {"rows_before": before, "rows_after": cleaned.height}


def clean_oil(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    before = df.height
    cleaned = (
        df.sort("date")
        .unique(subset=["date"], keep="first")
        .with_columns(pl.col("dcoilwtico").fill_null(strategy="forward"))
    )
    return cleaned, {"rows_before": before, "rows_after": cleaned.height}


def clean_holidays(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    before = df.height
    cleaned = normalize_text_columns(df).filter(~pl.col("transferred")).unique()
    return cleaned, {"rows_before": before, "rows_after": cleaned.height}


def clean_transactions(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    before = df.height
    cleaned = (
        df.filter((pl.col("transactions").is_not_null()) & (pl.col("transactions") >= 0))
        .group_by(["date", "store_nbr"])
        .agg(pl.col("transactions").sum())
    )
    return cleaned, {"rows_before": before, "rows_after": cleaned.height}


def clean_raw_tables(
    tables: dict[str, pl.DataFrame],
) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    cleaners = {
        "train": clean_train,
        "stores": clean_stores,
        "oil": clean_oil,
        "holidays_events": clean_holidays,
        "transactions": clean_transactions,
    }
    cleaned_tables: dict[str, pl.DataFrame] = {}
    metrics: dict[str, Any] = {}
    for name, cleaner in cleaners.items():
        cleaned_tables[name], metrics[name] = cleaner(tables[name])
    return cleaned_tables, metrics
