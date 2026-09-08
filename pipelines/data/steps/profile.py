from __future__ import annotations

from typing import Any

import polars as pl
from zenml import step
from zenml.logger import get_logger

from mlops_project.data.profiling import profile_tables

logger = get_logger(__name__)


@step
def profile_data(tables: dict[str, pl.DataFrame]) -> dict[str, Any]:
    profile = profile_tables(tables)
    logger.info("[profile] tables=%s", sorted(profile))
    return profile
