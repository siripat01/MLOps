from __future__ import annotations

from typing import Annotated, Any

import polars as pl
from zenml import step
from zenml.logger import get_logger

from mlops_project.config.settings import get_settings
from mlops_project.data.cleaning import clean_raw_tables
from mlops_project.data.quality import validate_cleaning_row_loss
from mlops_project.storage.object_store import write_parquet_if_configured

logger = get_logger(__name__)


@step
def clean_data(tables: dict[str, pl.DataFrame]) -> tuple[
    Annotated[dict[str, pl.DataFrame], "cleaned_tables"],
    Annotated[dict[str, Any], "cleaning_metadata"],
]:
    settings = get_settings()
    cleaned, metrics = clean_raw_tables(tables)
    quality_report = validate_cleaning_row_loss(metrics)
    metrics["quality_report"] = quality_report
    for name, df in cleaned.items():
        uri = settings.intermediate_uri(f"clean/{name}")
        write_parquet_if_configured(df, uri, settings)
        metrics[name]["uri"] = uri
        logger.info(
            "[clean] table=%s rows_before=%s rows_after=%s",
            name,
            metrics[name]["rows_before"],
            metrics[name]["rows_after"],
        )
    return cleaned, metrics
