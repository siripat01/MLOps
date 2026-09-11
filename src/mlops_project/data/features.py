from __future__ import annotations

import polars as pl

LAG_WINDOWS = (1, 7, 14, 28)
ROLLING_WINDOWS = (7, 14, 28)
OIL_LAG_WINDOWS = (1, 7)
KNOWN_COVARIATE_COLUMNS = [
    "onpromotion",
    "is_onpromotion",
    "promotion_log1p",
    "promotion_capped_10",
    "is_holiday",
    "holiday_count",
    "is_payday",
    "is_month_end",
    "is_quarter_end",
    "is_year_end",
    "year",
    "month",
    "quarter",
    "week_of_year",
    "day_of_week",
    "day_of_month",
    "is_weekend",
    "holiday_type_BRIDGE",
    "holiday_type_EVENT",
    "holiday_type_HOLIDAY",
    "holiday_type_TRANSFER",
    "holiday_type_WORK_DAY",
]
STATIC_COVARIATE_COLUMNS = [
    "store_nbr",
    "family",
    "city",
    "state",
    "store_type",
    "cluster",
]
HISTORICAL_COVARIATE_COLUMNS = [
    *[f"sales_lag_{window}" for window in LAG_WINDOWS],
    *[f"sales_rolling_mean_{window}" for window in ROLLING_WINDOWS],
    *[f"sales_rolling_std_{window}" for window in ROLLING_WINDOWS],
    *[f"transactions_lag_{window}" for window in LAG_WINDOWS],
    *[f"transactions_rolling_mean_{window}" for window in ROLLING_WINDOWS],
    *[f"transactions_rolling_std_{window}" for window in ROLLING_WINDOWS],
    "store_sales_lag_1",
    "store_sales_rolling_mean_7",
    "family_sales_lag_1",
    "family_sales_rolling_mean_7",
    "oil_lag_1",
    "oil_lag_7",
    "oil_rolling_mean_7",
]
TRAINING_COLUMNS = ["item_id", "date", "sales", *KNOWN_COVARIATE_COLUMNS]


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


def _aggregate_history_features(
    df: pl.DataFrame,
    group_keys: list[str],
    value_name: str,
    prefix: str,
) -> pl.DataFrame:
    return (
        df.group_by([*group_keys, "date"])
        .agg(pl.col(value_name).sum().alias(f"_{prefix}_daily_total"))
        .sort([*group_keys, "date"])
        .with_columns(
            pl.col(f"_{prefix}_daily_total").shift(1).over(group_keys).alias(f"{prefix}_lag_1"),
            pl.col(f"_{prefix}_daily_total")
            .shift(1)
            .rolling_mean(7)
            .over(group_keys)
            .alias(f"{prefix}_rolling_mean_7"),
        )
        .select([*group_keys, "date", f"{prefix}_lag_1", f"{prefix}_rolling_mean_7"])
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
        pl.col("date").dt.day().is_in([15]).alias("is_payday"),
        pl.col("date").dt.month_end().eq(pl.col("date")).alias("is_month_end"),
        (
            pl.col("date").dt.month().is_in([3, 6, 9, 12])
            & pl.col("date").dt.month_end().eq(pl.col("date"))
        ).alias("is_quarter_end"),
        pl.col("date").dt.strftime("%m-%d").eq("12-31").alias("is_year_end"),
        (pl.col("onpromotion") > 0).alias("is_onpromotion"),
        pl.col("onpromotion").cast(pl.Float64).log1p().alias("promotion_log1p"),
        pl.min_horizontal(pl.col("onpromotion"), pl.lit(10)).alias("promotion_capped_10"),
        (pl.col("holiday_type") == "BRIDGE").alias("holiday_type_BRIDGE"),
        (pl.col("holiday_type") == "EVENT").alias("holiday_type_EVENT"),
        (pl.col("holiday_type") == "HOLIDAY").alias("holiday_type_HOLIDAY"),
        (pl.col("holiday_type") == "TRANSFER").alias("holiday_type_TRANSFER"),
        (pl.col("holiday_type") == "WORK DAY").alias("holiday_type_WORK_DAY"),
        *historical_exprs,
    )

    # Oil and transactions are observed signals, not guaranteed future-known covariates.
    # Keep only lagged/rolling versions in the model-facing dataset.
    return (
        featured.join(_oil_history_features(featured), on="date", how="left")
        .join(
            _aggregate_history_features(featured, ["store_nbr"], "sales", "store_sales"),
            on=["store_nbr", "date"],
            how="left",
        )
        .join(
            _aggregate_history_features(featured, ["family"], "sales", "family_sales"),
            on=["family", "date"],
            how="left",
        )
        .drop(["transactions", "dcoilwtico"])
        .sort([*group_keys, "date"])
    )


def feature_names(df: pl.DataFrame) -> list[str]:
    non_features = {"id", "sales", "date"}
    return [col for col in df.columns if col not in non_features]


def feature_column_groups() -> dict[str, list[str]]:
    return {
        "training_columns": TRAINING_COLUMNS,
        "known_covariates": KNOWN_COVARIATE_COLUMNS,
        "static_covariates": STATIC_COVARIATE_COLUMNS,
        "historical_diagnostics": HISTORICAL_COVARIATE_COLUMNS,
    }
