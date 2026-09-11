from __future__ import annotations

from typing import Annotated, Any

import polars as pl
from zenml import step
from zenml.logger import get_logger

from mlops_project.config.settings import get_settings
from mlops_project.data.transformations import transform_store_sales
from mlops_project.storage.object_store import write_parquet_if_configured

logger = get_logger(__name__)


@step
def transform_data(df: pl.DataFrame) -> tuple[
    Annotated[pl.DataFrame, "transformed_data"],
    Annotated[dict[str, Any], "transform_metadata"],
]:
    settings = get_settings()
    transformed = transform_store_sales(df)
    uri = settings.intermediate_uri("transformed")
    write_parquet_if_configured(transformed, uri, settings)
    metadata = {"rows": transformed.height, "columns": transformed.width, "uri": uri}
    logger.info("[transform] rows=%s columns=%s", transformed.height, transformed.width)
    return transformed, metadata
