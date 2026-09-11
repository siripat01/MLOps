from __future__ import annotations

from typing import Annotated, Any

import polars as pl
from zenml import step
from zenml.logger import get_logger

from mlops_project.data.schemas.raw import validate_raw_tables

logger = get_logger(__name__)


@step
def validate_raw_data(
    tables: dict[str, pl.DataFrame],
    ingestion_metadata: dict[str, Any],
) -> tuple[
    Annotated[dict[str, pl.DataFrame], "validated_raw_tables"],
    Annotated[dict[str, Any], "raw_validation_metadata"],
]:
    validated = validate_raw_tables(tables)
    metadata = {
        "schema_valid": True,
        "source_metadata": ingestion_metadata,
        "tables": {
            name: {"rows": df.height, "columns": df.width}
            for name, df in validated.items()
        },
    }
    logger.info("[validate_raw] schema_valid=true tables=%s", sorted(validated))
    return validated, metadata
