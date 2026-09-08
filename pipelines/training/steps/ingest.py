import polars as pl
from zenml import step

from mlops_project.data.ingest import IngestDataFromKaggle


@step
def IngestData() -> pl.DataFrame:
    df = IngestDataFromKaggle()
    return df
