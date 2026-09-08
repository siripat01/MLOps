from __future__ import annotations

import os
from typing import Annotated, Any

import polars as pl
from zenml import ArtifactConfig, step
from zenml.enums import ArtifactType
from zenml.logger import get_logger

from mlops_project.data.schemas.features import validate_feature_table

logger = get_logger(__name__)

FEATURE_ARTIFACT_NAME = os.getenv("FEATURE_ARTIFACT_NAME", "store_sales_features")
# Leave unset to let ZenML assign versions "1", "2", "3", ...
ZENML_FEATURE_VERSION = (
    os.getenv("ZENML_FEATURE_VERSION")
    or os.getenv("FEATURE_VERSION")  # backwards compatibility
    or None
)
FEATURE_METADATA_ARTIFACT_NAME = f"{FEATURE_ARTIFACT_NAME}_metadata"


@step
def validate_features(
    features: pl.DataFrame, feature_metadata: dict[str, Any]
) -> tuple[
    Annotated[
        pl.DataFrame,
        ArtifactConfig(
            name=FEATURE_ARTIFACT_NAME,
            version=ZENML_FEATURE_VERSION,
            artifact_type=ArtifactType.DATA,
            tags=["features", "store-sales"],
        ),
    ],
    Annotated[
        dict[str, Any],
        ArtifactConfig(
            name=FEATURE_METADATA_ARTIFACT_NAME,
            version=ZENML_FEATURE_VERSION,
            tags=["features", "metadata", "store-sales"],
        ),
    ],
]:
    validated = validate_feature_table(features)
    null_percentages = {
        col: validated.select(pl.col(col).is_null().mean()).item() for col in validated.columns
    }
    high_missing = {col: pct for col, pct in null_percentages.items() if pct > 0.95}
    if high_missing:
        raise ValueError(f"Features with >95% missing values: {high_missing}")
    metadata = {
        **feature_metadata,
        "schema_valid": True,
        "high_missing_features": high_missing,
        "zenml_version_policy": "custom" if ZENML_FEATURE_VERSION else "auto_increment",
    }
    logger.info(
        "[validate_features] schema_valid=true rows=%s artifact=%s version_policy=%s",
        validated.height,
        FEATURE_ARTIFACT_NAME,
        metadata["zenml_version_policy"],
    )
    return validated, metadata
