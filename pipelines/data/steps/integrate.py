from __future__ import annotations

from typing import Annotated, Any

import polars as pl
from zenml import log_metadata, step
from zenml.logger import get_logger

from mlops_project.config.settings import get_settings
from mlops_project.data.integration import integrate_store_sales
from mlops_project.storage.object_store import write_parquet_if_configured

logger = get_logger(__name__)


@step
def integrate_data(tables: dict[str, pl.DataFrame]) -> tuple[
    Annotated[pl.DataFrame, "integrated_data"],
    Annotated[dict[str, Any], "integration_metadata"],
]:
    settings = get_settings()
    integrated, metrics = integrate_store_sales(tables)
    uri = settings.intermediate_uri("integrated")
    write_parquet_if_configured(integrated, uri, settings)
    metrics["uri"] = uri
    log_metadata(
        metadata={
            "rows_before": metrics["rows_before"],
            "rows_after": metrics["rows_after"],
            "store_join_miss_rate": metrics["store_join_miss_rate"],
            "raw_oil_join_miss_rate": metrics["raw_oil_join_miss_rate"],
            "oil_join_miss_rate": metrics["oil_join_miss_rate"],
            "transaction_join_miss_rate": metrics["transaction_join_miss_rate"],
            "row_multiplication_factor": metrics["row_multiplication_factor"],
        }
    )
    logger.info(
        "[join] rows_before=%s rows_after=%s unmatched_stores=%s",
        metrics["rows_before"],
        metrics["rows_after"],
        metrics["unmatched_stores"],
    )
    return integrated, metrics
