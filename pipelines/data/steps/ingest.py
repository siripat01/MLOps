from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
from zenml import step
from zenml.logger import get_logger

from mlops_project.config.settings import get_settings
from mlops_project.storage.object_store import write_parquet_if_configured

logger = get_logger(__name__)

REQUIRED_RAW_FILES = {
    "train": "train.csv",
    "stores": "stores.csv",
    "oil": "oil.csv",
    "holidays_events": "holidays_events.csv",
    "transactions": "transactions.csv",
}


def load_store_sales_from_kaggle(competition: str) -> dict[str, pl.DataFrame]:
    import kagglehub

    path = Path(kagglehub.competition_download(competition))
    tables: dict[str, pl.DataFrame] = {}
    for name, filename in REQUIRED_RAW_FILES.items():
        tables[name] = pl.read_csv(path / filename, try_parse_dates=True)
    return tables


@step(enable_cache=False)
def ingest_data() -> tuple[dict[str, pl.DataFrame], dict[str, Any]]:
    settings = get_settings()
    tables = load_store_sales_from_kaggle(settings.kaggle_competition)
    metadata: dict[str, Any] = {
        "source": "kaggle",
        "competition": settings.kaggle_competition,
        "dataset_name": settings.dataset_name,
        "dataset_version": settings.dataset_version,
        "ingestion_timestamp": datetime.now(UTC).isoformat(),
        "tables": {},
    }
    for name, df in tables.items():
        uri = None
        if settings.raw_uri_prefix:
            uri = f"{settings.raw_uri_prefix}/{name}.parquet"
            write_parquet_if_configured(df, uri, settings)
        metadata["tables"][name] = {
            "rows": df.height,
            "columns": df.width,
            "column_names": df.columns,
            "uri": uri,
        }
        logger.info("[ingest] table=%s rows=%s columns=%s uri=%s", name, df.height, df.width, uri)
    return tables, metadata
