from __future__ import annotations

from typing import Annotated, Any

import polars as pl
from zenml import step
from zenml.logger import get_logger

from mlops_project.data.quality import series_calendar_report
from mlops_project.data.schemas.cleaned import validate_cleaned_tables

logger = get_logger(__name__)


@step
def validate_cleaned_data(
    tables: dict[str, pl.DataFrame],
) -> tuple[
    Annotated[dict[str, pl.DataFrame], "validated_cleaned_tables"],
    Annotated[dict[str, Any], "cleaned_validation_metadata"],
]:
    validated = validate_cleaned_tables(tables)
    metadata = {
        "schema_valid": True,
        "tables": {
            name: {"rows": df.height, "columns": df.width}
            for name, df in validated.items()
        },
        "forecast_grain": "store_nbr+family+date",
        "train_series_quality": series_calendar_report(validated["train"]),
    }
    logger.info("[validate_cleaned] schema_valid=true tables=%s", sorted(validated))
    return validated, metadata
