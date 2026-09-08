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


def _metrics(before: int, cleaned: pl.DataFrame) -> dict[str, int]:
    return {
        "rows_before": before,
        "rows_after": cleaned.height,
        "rows_removed": before - cleaned.height,
    }


def clean_train(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    before = df.height
    cleaned = (
        normalize_text_columns(df)
        .unique(subset=["id"], keep="first", maintain_order=True)
        .filter(
            pl.col("id").is_not_null()
            & pl.col("date").is_not_null()
            & pl.col("store_nbr").is_not_null()
            & (pl.col("store_nbr") > 0)
            & pl.col("family").is_not_null()
            & pl.col("sales").is_not_null()
            & (pl.col("sales") >= 0)
            & pl.col("onpromotion").is_not_null()
            & (pl.col("onpromotion") >= 0)
        )
    )
    return cleaned, _metrics(before, cleaned)


def clean_stores(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    before = df.height
    cleaned = (
        normalize_text_columns(df)
        .filter(
            pl.col("store_nbr").is_not_null()
            & (pl.col("store_nbr") > 0)
            & pl.col("city").is_not_null()
            & pl.col("state").is_not_null()
            & pl.col("type").is_not_null()
            & pl.col("cluster").is_not_null()
            & (pl.col("cluster") > 0)
        )
        .unique(subset=["store_nbr"], keep="first", maintain_order=True)
    )
    return cleaned, _metrics(before, cleaned)


def clean_oil(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    before = df.height
    cleaned = (
        df.filter(
            pl.col("date").is_not_null()
            & (pl.col("dcoilwtico").is_null() | (pl.col("dcoilwtico") >= 0))
        )
        .sort("date")
        .unique(subset=["date"], keep="first", maintain_order=True)
        .with_columns(pl.col("dcoilwtico").fill_null(strategy="forward"))
    )
    return cleaned, _metrics(before, cleaned)


def clean_holidays(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    before = df.height
    normalized = normalize_text_columns(df)
    cleaned = (
        normalized.filter(
            pl.col("date").is_not_null()
            & pl.col("type").is_not_null()
            & pl.col("locale").is_not_null()
            & pl.col("locale_name").is_not_null()
            & pl.col("description").is_not_null()
            & pl.col("transferred").is_not_null()
            & (~pl.col("transferred"))
        )
        .filter(pl.col("locale").is_in(["NATIONAL", "REGIONAL", "LOCAL"]))
        .unique(maintain_order=True)
    )
    return cleaned, _metrics(before, cleaned)


def clean_transactions(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    before = df.height
    cleaned = (
        df.filter(
            pl.col("date").is_not_null()
            & pl.col("store_nbr").is_not_null()
            & (pl.col("store_nbr") > 0)
            & pl.col("transactions").is_not_null()
            & (pl.col("transactions") >= 0)
        )
        .group_by(["date", "store_nbr"])
        .agg(pl.col("transactions").sum())
        .sort(["store_nbr", "date"])
    )
    return cleaned, _metrics(before, cleaned)


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
