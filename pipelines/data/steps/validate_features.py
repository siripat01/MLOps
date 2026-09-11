from __future__ import annotations

import os
from typing import Annotated, Any

import polars as pl
from zenml import ArtifactConfig, log_metadata, step
from zenml.enums import ArtifactType
from zenml.logger import get_logger

from mlops_project.data.features import TRAINING_COLUMNS, feature_column_groups
from mlops_project.data.quality import missingness_report, series_calendar_report
from mlops_project.data.schemas.features import validate_feature_table

logger = get_logger(__name__)

FEATURE_ARTIFACT_NAME = os.getenv("FEATURE_ARTIFACT_NAME", "store_sales_features")
FEATURE_METADATA_ARTIFACT_NAME = f"{FEATURE_ARTIFACT_NAME}_metadata"


@step
def validate_features(
    features: pl.DataFrame, feature_metadata: dict[str, Any]
) -> tuple[
    Annotated[
        pl.DataFrame,
        ArtifactConfig(
            name=FEATURE_ARTIFACT_NAME,
            artifact_type=ArtifactType.DATA,
            tags=["features", "store-sales"],
        ),
    ],
    Annotated[
        dict[str, Any],
        ArtifactConfig(
            name=FEATURE_METADATA_ARTIFACT_NAME,
            tags=["features", "metadata", "store-sales"],
        ),
    ],
]:
    validated = validate_feature_table(features)
    null_percentages = missingness_report(validated, validated.columns)
    high_missing = {col: pct for col, pct in null_percentages.items() if pct > 0.95}
    if high_missing:
        raise ValueError(f"Features with >95% missing values: {high_missing}")
    training_missingness = missingness_report(validated, TRAINING_COLUMNS)
    metadata = {
        **feature_metadata,
        "schema_valid": True,
        "forecast_grain": "store_nbr+family+date",
        "quality_summary": {
            "rows": validated.height,
            "columns": validated.width,
            "series": series_calendar_report(validated),
            "training_missingness": training_missingness,
        },
        "feature_column_groups": feature_column_groups(),
        "training_columns": TRAINING_COLUMNS,
        "high_missing_features": high_missing,
        "zenml_version_policy": "auto_increment",
    }
    log_metadata(
        metadata={
            "rows": validated.height,
            "columns": validated.width,
            "forecast_grain": metadata["forecast_grain"],
            "feature_data_version": metadata.get("feature_data_version"),
            "schema_version": metadata.get("schema_version"),
            "series_count": metadata["quality_summary"]["series"]["series_count"],
        }
    )
    logger.info(
        "[validate_features] schema_valid=true rows=%s artifact=%s version_policy=auto_increment",
        validated.height,
        FEATURE_ARTIFACT_NAME,
    )
    return validated, metadata
