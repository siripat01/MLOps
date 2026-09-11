from __future__ import annotations

from typing import Any

import polars as pl

FORECAST_GRAIN = ["store_nbr", "family", "date"]
SERIES_KEYS = ["store_nbr", "family"]


def duplicate_grain_count(df: pl.DataFrame) -> int:
    return df.select(pl.struct(FORECAST_GRAIN).is_duplicated().sum()).item()


def assert_unique_forecast_grain(df: pl.DataFrame, table_name: str) -> None:
    duplicates = duplicate_grain_count(df)
    if duplicates:
        raise ValueError(
            f"{table_name} has duplicate store/family/date rows: {duplicates}"
        )


def series_calendar_report(df: pl.DataFrame) -> dict[str, Any]:
    if df.is_empty():
        return {
            "series_count": 0,
            "dataset_start": None,
            "dataset_end": None,
            "min_observations_per_series": 0,
            "max_observations_per_series": 0,
            "missing_calendar_dates": 0,
            "global_calendar_gap_dates": 0,
            "incomplete_series_count": 0,
        }

    dataset_calendar = df.select("date").unique().sort("date")
    dataset_start = dataset_calendar.select(pl.col("date").min()).item()
    dataset_end = dataset_calendar.select(pl.col("date").max()).item()
    calendar_days = (
        pl.DataFrame({"date": pl.date_range(dataset_start, dataset_end, eager=True)})
        if dataset_start is not None and dataset_end is not None
        else pl.DataFrame({"date": []}, schema={"date": pl.Date})
    )
    global_calendar_gap_dates = calendar_days.join(
        dataset_calendar,
        on="date",
        how="anti",
    ).height

    coverage = (
        df.group_by(SERIES_KEYS)
        .agg(
            pl.col("date").min().alias("start_date"),
            pl.col("date").max().alias("end_date"),
            pl.col("date").n_unique().alias("observations"),
        )
        .join_where(
            dataset_calendar.rename({"date": "_calendar_date"}),
            pl.col("_calendar_date").is_between(
                pl.col("start_date"),
                pl.col("end_date"),
            ),
        )
        .group_by([*SERIES_KEYS, "start_date", "end_date", "observations"])
        .agg(pl.col("_calendar_date").n_unique().alias("expected_observations"))
        .with_columns(
            (pl.col("expected_observations") - pl.col("observations")).alias(
                "missing_dates"
            )
        )
    )

    return {
        "series_count": coverage.height,
        "dataset_start": str(df.select(pl.col("date").min()).item()),
        "dataset_end": str(df.select(pl.col("date").max()).item()),
        "min_observations_per_series": coverage.select(
            pl.col("observations").min()
        ).item(),
        "max_observations_per_series": coverage.select(
            pl.col("observations").max()
        ).item(),
        "missing_calendar_dates": coverage.select(pl.col("missing_dates").sum()).item(),
        "global_calendar_gap_dates": global_calendar_gap_dates,
        "incomplete_series_count": coverage.filter(pl.col("missing_dates") > 0).height,
    }


def validate_series_calendar(
    df: pl.DataFrame,
    *,
    table_name: str,
    min_history_per_series: int = 1,
    max_missing_calendar_dates: int = 0,
) -> dict[str, Any]:
    assert_unique_forecast_grain(df, table_name)
    report = series_calendar_report(df)

    if report["min_observations_per_series"] < min_history_per_series:
        raise ValueError(
            f"{table_name} has series with insufficient history: "
            f"minimum_required={min_history_per_series}, "
            f"observed_min={report['min_observations_per_series']}"
        )

    if report["missing_calendar_dates"] > max_missing_calendar_dates:
        raise ValueError(
            f"{table_name} has missing calendar dates within item series: "
            f"allowed={max_missing_calendar_dates}, "
            f"observed={report['missing_calendar_dates']}"
        )

    return report


def cleaning_quality_report(metrics: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for table_name, table_metrics in metrics.items():
        rows_before = table_metrics.get("rows_before", 0)
        rows_removed = table_metrics.get("rows_removed", 0)
        report[table_name] = {
            "rows_before": rows_before,
            "rows_after": table_metrics.get("rows_after", 0),
            "rows_removed": rows_removed,
            "row_loss_rate": rows_removed / rows_before if rows_before else 0.0,
        }
    return report


def validate_cleaning_row_loss(
    metrics: dict[str, Any],
    *,
    max_loss_rates: dict[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = max_loss_rates or {
        "train": 0.10,
        "stores": 0.02,
        "oil": 0.25,
        "holidays_events": 0.80,
        "transactions": 0.10,
    }
    report = cleaning_quality_report(metrics)
    violations = {
        table_name: {
            "row_loss_rate": table_report["row_loss_rate"],
            "max_loss_rate": thresholds[table_name],
        }
        for table_name, table_report in report.items()
        if table_name in thresholds
        and table_report["row_loss_rate"] > thresholds[table_name]
    }
    if violations:
        raise ValueError(f"Unexpected row loss during cleaning: {violations}")
    return report


def missingness_report(df: pl.DataFrame, columns: list[str]) -> dict[str, float]:
    return {
        col: df.select(pl.col(col).is_null().mean()).item()
        for col in columns
        if col in df.columns
    }
