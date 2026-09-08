from __future__ import annotations

import polars as pl

LAG_WINDOWS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 14, 28)


def engineer_store_sales_features(df: pl.DataFrame) -> pl.DataFrame:
    sorted_df = df.sort(["store_nbr", "family", "date"])
    lag_exprs = [
        pl.col("sales").shift(window).over(["store_nbr", "family"]).alias(f"lag_{window}")
        for window in LAG_WINDOWS
    ]
    rolling_exprs = []
    for window in ROLLING_WINDOWS:
        historical_sales = pl.col("sales").shift(1).over(["store_nbr", "family"])
        rolling_exprs.extend(
            [
                historical_sales.rolling_mean(window).over(["store_nbr", "family"]).alias(
                    f"rolling_mean_{window}"
                ),
                historical_sales.rolling_std(window).over(["store_nbr", "family"]).alias(
                    f"rolling_std_{window}"
                ),
            ]
        )
    return sorted_df.with_columns(
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.quarter().alias("quarter"),
        pl.col("date").dt.week().alias("week_of_year"),
        pl.col("date").dt.weekday().alias("day_of_week"),
        pl.col("date").dt.day().alias("day_of_month"),
        (pl.col("date").dt.weekday() >= 6).alias("is_weekend"),
        *lag_exprs,
        *rolling_exprs,
    )


def feature_names(df: pl.DataFrame) -> list[str]:
    non_features = {"id", "sales", "date"}
    return [col for col in df.columns if col not in non_features]
