from __future__ import annotations

from typing import Any

import polars as pl
from zenml import step
from zenml.logger import get_logger

from mlops_project.config.settings import get_settings
from mlops_project.data.integration import integrate_store_sales
from mlops_project.storage.object_store import write_parquet_if_configured

logger = get_logger(__name__)


@step
def integrate_data(tables: dict[str, pl.DataFrame]) -> tuple[pl.DataFrame, dict[str, Any]]:
    settings = get_settings()
    integrated, metrics = integrate_store_sales(tables)
    uri = settings.intermediate_uri("integrated")
    write_parquet_if_configured(integrated, uri, settings)
    metrics["uri"] = uri
    logger.info(
        "[join] rows_before=%s rows_after=%s unmatched_stores=%s",
        metrics["rows_before"],
        metrics["rows_after"],
        metrics["unmatched_stores"],
    )
    return integrated, metrics
