from __future__ import annotations

import pandas as pd
import polars as pl
from zenml import step

from mlops_project.features.covariates import training_frame_from_features


@step
def prepare_training_data(
    df: pl.DataFrame,
) -> pd.DataFrame:
    return training_frame_from_features(df)
