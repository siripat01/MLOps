from __future__ import annotations

import polars as pl

LAG_WINDOWS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 14, 28)
OIL_LAG_WINDOWS = (1, 7)


def _historical_group_features(
    column: str,
    prefix: str,
    group_keys: list[str],
) -> list[pl.Expr]:
    lag_exprs = [
        pl.col(column).shift(window).over(group_keys).alias(f"{prefix}_lag_{window}")
        for window in LAG_WINDOWS
    ]
    rolling_exprs: list[pl.Expr] = []
    for window in ROLLING_WINDOWS:
        rolling_exprs.extend(
            [
                pl.col(column)
                .shift(1)
                .rolling_mean(window)
                .over(group_keys)
                .alias(f"{prefix}_rolling_mean_{window}"),
                pl.col(column)
                .shift(1)
                .rolling_std(window)
                .over(group_keys)
                .alias(f"{prefix}_rolling_std_{window}"),
            ]
        )
    return [*lag_exprs, *rolling_exprs]


def _oil_history_features(df: pl.DataFrame) -> pl.DataFrame:
    oil_by_date = (
        df.select(["date", "dcoilwtico"])
        .unique(subset=["date"], keep="first", maintain_order=True)
        .sort("date")
        .with_columns(
            pl.col("dcoilwtico").fill_null(strategy="forward").alias("_oil_observed")
        )
    )

    lag_exprs = [
        pl.col("_oil_observed").shift(window).alias(f"oil_lag_{window}")
        for window in OIL_LAG_WINDOWS
    ]
    return (
        oil_by_date.with_columns(
            *lag_exprs,
            pl.col("_oil_observed").shift(1).rolling_mean(7).alias("oil_rolling_mean_7"),
        )
        .select(
            "date",
            *[f"oil_lag_{window}" for window in OIL_LAG_WINDOWS],
            "oil_rolling_mean_7",
        )
    )


def engineer_store_sales_features(df: pl.DataFrame) -> pl.DataFrame:
    group_keys = ["store_nbr", "family"]
    sorted_df = df.sort([*group_keys, "date"])

    historical_exprs = [
        *_historical_group_features("sales", "sales", group_keys),
        *_historical_group_features("transactions", "transactions", group_keys),
    ]

    featured = sorted_df.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.quarter().alias("quarter"),
        pl.col("date").dt.week().alias("week_of_year"),
        pl.col("date").dt.weekday().alias("day_of_week"),
        pl.col("date").dt.day().alias("day_of_month"),
        (pl.col("date").dt.weekday() >= 6).alias("is_weekend"),
        *historical_exprs,
    )

    # Oil and transactions are observed signals, not guaranteed future-known covariates.
    # Keep only lagged/rolling versions in the model-facing dataset.
    return (
        featured.join(_oil_history_features(featured), on="date", how="left")
        .drop(["transactions", "dcoilwtico"])
        .sort([*group_keys, "date"])
    )


def feature_names(df: pl.DataFrame) -> list[str]:
    non_features = {"id", "sales", "date"}
    return [col for col in df.columns if col not in non_features]
