from __future__ import annotations

import pandas as pd
import polars as pl
from zenml import step


@step
def prepare_training_data(
    df: pl.DataFrame,
) -> pd.DataFrame:
    return (
        df.select(
            [
                "item_id",
                "date",
                "sales",
                "onpromotion",
                "is_holiday",
            ]
        )
        .to_pandas()
    )
