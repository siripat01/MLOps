# Data

The current pipeline uses Kaggle's Store Sales time-series competition as the
active source. Local files under this directory are legacy artifacts and are not
used by the ZenML data pipeline.

Run the active feature pipeline with:

```bash
uv run run-data-pipeline
```

ZenML stores returned step artifacts in the active artifact store. For this
project's local stack, that artifact store is MinIO at `s3://zenml`.
