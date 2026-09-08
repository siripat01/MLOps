from __future__ import annotations

from typing import Any

import polars as pl


def profile_table(df: pl.DataFrame, top_k: int = 5) -> dict[str, Any]:
    rows = df.height
    null_counts = dict(zip(df.columns, df.null_count().row(0), strict=True))
    profile: dict[str, Any] = {
        "rows": rows,
        "columns": df.width,
        "column_names": df.columns,
        "null_counts": null_counts,
        "null_percentage": {
            col: (count / rows if rows else 0.0) for col, count in null_counts.items()
        },
        "unique_counts": {col: df.select(pl.col(col).n_unique()).item() for col in df.columns},
        "duplicates": df.is_duplicated().sum(),
        "estimated_size_bytes": df.estimated_size(),
    }
    numeric_stats: dict[str, dict[str, float | int | None]] = {}
    categorical_frequencies: dict[str, list[dict[str, Any]]] = {}
    for col, dtype in df.schema.items():
        if dtype.is_numeric():
            stats = df.select(
                pl.col(col).min().alias("min"),
                pl.col(col).max().alias("max"),
                pl.col(col).mean().alias("mean"),
                pl.col(col).median().alias("median"),
            ).row(0, named=True)
            numeric_stats[col] = stats
        elif dtype == pl.String:
            counts = (
                df.group_by(col)
                .len(name="count")
                .sort("count", descending=True)
                .head(top_k)
                .to_dicts()
            )
            categorical_frequencies[col] = counts
    profile["numeric_stats"] = numeric_stats
    profile["categorical_frequencies"] = categorical_frequencies
    return profile


def profile_tables(tables: dict[str, pl.DataFrame]) -> dict[str, Any]:
    return {name: profile_table(df) for name, df in tables.items()}
