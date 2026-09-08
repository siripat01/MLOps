from __future__ import annotations

from datetime import timedelta
from typing import Any

import polars as pl
from zenml import step


@step
def split_data(
    df: pl.DataFrame,
    dataset_metadata: dict[str, Any],
    prediction_length: int = 16,
    min_train_observations: int = 60,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    dict[str, Any],
]:
    """Split a time-series dataset into train and validation sets.

    The last `prediction_length` calendar days are reserved for validation.
    This simulates a real forecasting origin and avoids random-split leakage.
    """

    required_columns = {"item_id", "date", "sales"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns for time-series split: "
            f"{sorted(missing_columns)}"
        )

    if prediction_length <= 0:
        raise ValueError("prediction_length must be greater than 0")

    df = df.sort(["item_id", "date"])

    # Each item should have at most one observation per date.
    duplicate_rows = (
        df.group_by(["item_id", "date"])
        .len()
        .filter(pl.col("len") > 1)
    )

    if duplicate_rows.height > 0:
        raise ValueError(
            "Duplicate (item_id, date) rows found before train/validation split"
        )

    max_date = df.select(
        pl.col("date").max()
    ).item()

    min_date = df.select(
        pl.col("date").min()
    ).item()

    if max_date is None or min_date is None:
        raise ValueError("Dataset has no valid dates")

    validation_start = max_date - timedelta(
        days=prediction_length - 1
    )

    train_df = df.filter(
        pl.col("date") < validation_start
    )

    validation_df = df.filter(
        pl.col("date") >= validation_start
    )

    if train_df.is_empty():
        raise ValueError(
            "Training split is empty. "
            "Reduce prediction_length or provide more historical data."
        )

    if validation_df.is_empty():
        raise ValueError("Validation split is empty")

    # Make sure validation really contains the requested number of dates.
    validation_date_count = validation_df.select(
        pl.col("date").n_unique()
    ).item()

    if validation_date_count != prediction_length:
        raise ValueError(
            "Validation horizon does not contain the expected number "
            f"of calendar dates: expected={prediction_length}, "
            f"actual={validation_date_count}"
        )

    # Every series should have the full validation horizon.
    validation_coverage = (
        validation_df.group_by("item_id")
        .agg(
            pl.col("date")
            .n_unique()
            .alias("validation_observations")
        )
    )

    incomplete_validation = validation_coverage.filter(
        pl.col("validation_observations")
        != prediction_length
    )

    if incomplete_validation.height > 0:
        sample = incomplete_validation.head(10).to_dicts()

        raise ValueError(
            "Some item_id series do not contain the full "
            f"{prediction_length}-day validation horizon. "
            f"Examples: {sample}"
        )

    # Ensure every series has enough training history.
    train_coverage = (
        train_df.group_by("item_id")
        .agg(
            pl.len().alias("train_observations")
        )
    )

    insufficient_history = train_coverage.filter(
        pl.col("train_observations")
        < min_train_observations
    )

    if insufficient_history.height > 0:
        sample = insufficient_history.head(10).to_dicts()

        raise ValueError(
            "Some item_id series do not have enough training history. "
            f"Minimum required={min_train_observations}. "
            f"Examples: {sample}"
        )

    train_items = train_df["item_id"].n_unique()
    validation_items = validation_df["item_id"].n_unique()

    if train_items != validation_items:
        raise ValueError(
            "Train and validation contain a different number of item_id "
            f"series: train={train_items}, "
            f"validation={validation_items}"
        )

    metadata = {
        "source_artifact_name": dataset_metadata.get(
            "artifact_name"
        ),
        "source_artifact_version": dataset_metadata.get(
            "artifact_version"
        ),
        "prediction_length": prediction_length,
        "dataset_start": str(min_date),
        "dataset_end": str(max_date),
        "train_end": str(
            validation_start - timedelta(days=1)
        ),
        "validation_start": str(validation_start),
        "validation_end": str(max_date),
        "train_rows": train_df.height,
        "validation_rows": validation_df.height,
        "train_items": train_items,
        "validation_items": validation_items,
    }

    return train_df, validation_df, metadata
