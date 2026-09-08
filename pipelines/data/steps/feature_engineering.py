from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import polars as pl
from zenml import step
from zenml.logger import get_logger

from mlops_project.config.settings import get_settings
from mlops_project.data.features import engineer_store_sales_features, feature_names
from mlops_project.storage.object_store import write_parquet_if_configured

logger = get_logger(__name__)


@step
def engineer_features(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, Any]]:
    settings = get_settings()
    features = engineer_store_sales_features(df)
    uri = settings.feature_uri
    write_parquet_if_configured(features, uri, settings)
    metadata = {
        "dataset_name": settings.dataset_name,
        "dataset_version": settings.dataset_version,
        "feature_version": settings.feature_version,
        "schema_version": "features.v1",
        "generation_timestamp": datetime.now(UTC).isoformat(),
        "rows": features.height,
        "columns": features.width,
        "feature_names": feature_names(features),
        "uri": uri,
    }
    logger.info("[features] rows=%s features=%s", features.height, len(metadata["feature_names"]))
    return features, metadata
