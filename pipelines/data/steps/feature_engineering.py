from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any

import polars as pl
from zenml import step
from zenml.logger import get_logger

from mlops_project.config.settings import get_settings
from mlops_project.data.features import (
    engineer_store_sales_features,
    feature_column_groups,
    feature_names,
)
from mlops_project.data.quality import missingness_report, series_calendar_report
from mlops_project.storage.object_store import write_parquet_if_configured

logger = get_logger(__name__)


@step
def engineer_features(df: pl.DataFrame) -> tuple[
    Annotated[pl.DataFrame, "engineered_features"],
    Annotated[dict[str, Any], "feature_engineering_metadata"],
]:
    settings = get_settings()
    features = engineer_store_sales_features(df)
    uri = settings.feature_uri
    write_parquet_if_configured(features, uri, settings)
    metadata = {
        "dataset_name": settings.dataset_name,
        "dataset_version": settings.dataset_version,
        "feature_data_version": settings.resolved_feature_data_version,
        "zenml_artifact_name": settings.feature_artifact_name,
        "schema_version": "features.v2",
        "forecast_grain": "store_nbr+family+date",
        "generation_timestamp": datetime.now(UTC).isoformat(),
        "rows": features.height,
        "columns": features.width,
        "feature_names": feature_names(features),
        "feature_column_groups": feature_column_groups(),
        "series_quality": series_calendar_report(features),
        "missingness": missingness_report(features, features.columns),
        "uri": uri,
    }
    logger.info(
        "[features] rows=%s features=%s uri=%s",
        features.height,
        len(metadata["feature_names"]),
        uri,
    )
    return features, metadata
