from __future__ import annotations

import polars as pl


def transform_store_sales(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.concat_str(
                [pl.col("store_nbr").cast(pl.String), pl.col("family")], separator="_"
            ).alias("item_id"),
            pl.col("transactions").cast(pl.Int64),
            pl.col("dcoilwtico").fill_null(strategy="forward").over("store_nbr"),
        )
        .sort(["store_nbr", "family", "date"])
    )
