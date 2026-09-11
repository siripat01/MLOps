from __future__ import annotations

import pandas as pd
import polars as pl
from zenml import step

from mlops_project.data.features import TRAINING_COLUMNS


@step
def prepare_training_data(
    df: pl.DataFrame,
) -> pd.DataFrame:
    missing = sorted(set(TRAINING_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required training columns: {missing}")
    return df.select(TRAINING_COLUMNS).to_pandas()
