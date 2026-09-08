from __future__ import annotations

from typing import Any

import polars as pl
from zenml import step
from zenml.logger import get_logger

from mlops_project.data.schemas.features import validate_feature_table

logger = get_logger(__name__)


@step
def validate_features(
    features: pl.DataFrame, feature_metadata: dict[str, Any]
) -> tuple[pl.DataFrame, dict[str, Any]]:
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
    }
    logger.info("[validate_features] schema_valid=true rows=%s", validated.height)
    return validated, metadata
