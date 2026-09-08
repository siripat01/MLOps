from __future__ import annotations

import os

from zenml import pipeline

from pipelines.training.steps.ingest import IngestData


@pipeline
def load_data(artifact_version: str | None = None) -> None:
    IngestData(artifact_version=artifact_version)


def main() -> None:
    load_data(artifact_version=os.getenv("TRAIN_FEATURE_VERSION") or None)


if __name__ == "__main__":
    main()
