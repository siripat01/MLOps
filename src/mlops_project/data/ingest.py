from pathlib import Path

import polars as pl


def IngestDataFromKaggle(
    provider: str = "kaggle",
    dataset: str = "store-sales-time-series-forecasting",
) -> pl.DataFrame:
    import kagglehub

    if provider != "kaggle":
        raise ValueError(f"Unsupported data provider: {provider}")

    path = Path(kagglehub.competition_download(dataset))

    df = pl.read_csv(
        path / "train.csv",
        try_parse_dates=True,
    )

    return df


if __name__ == "__main__":
    print(IngestDataFromKaggle())
