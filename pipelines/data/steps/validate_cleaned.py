from __future__ import annotations

from typing import Any

import polars as pl
from zenml import step
from zenml.logger import get_logger

from mlops_project.data.schemas.cleaned import validate_cleaned_tables

logger = get_logger(__name__)


@step
def validate_cleaned_data(
    tables: dict[str, pl.DataFrame],
) -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    validated = validate_cleaned_tables(tables)
    metadata = {
        "schema_valid": True,
        "tables": {
            name: {"rows": df.height, "columns": df.width}
            for name, df in validated.items()
        },
    }
    logger.info("[validate_cleaned] schema_valid=true tables=%s", sorted(validated))
    return validated, metadata
