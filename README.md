# MLOps Project

A starter repository for reproducible model development, training, evaluation,
deployment, and monitoring.

## Structure

```text
.
├── configs/              # Versioned application and experiment configuration
├── data/                 # Local data by lifecycle stage (ignored by Git)
├── docs/                 # Architecture and operational documentation
├── infrastructure/       # IaC and deployment manifests
├── models/               # Local model artifacts (ignored by Git)
├── notebooks/            # Exploratory analysis
├── pipelines/            # Data, training, and deployment pipeline definitions
├── scripts/              # Developer and automation entry points
├── src/mlops_project/    # Reusable application code
└── tests/                # Unit and integration tests
```

## Quick start

```bash
uv sync --extra dev
uv run run-data-pipeline
uv run pytest
```

The project pins Python 3.12 in `.python-version`; `uv` manages the interpreter,
virtual environment, project dependencies, and lockfile.

## Data pipeline

The active source is Kaggle's Store Sales time-series competition. The data flow is:

```text
ingest_data
validate_raw_data          # structural contract only
profile_data
clean_data
validate_cleaned_data      # value-level contract
integrate_data             # locale-aware holiday joins
transform_data
engineer_features          # historical-only observed signals
validate_features
```

The final model-facing dataset intentionally excludes same-day `transactions` and
`dcoilwtico`. Historical lag/rolling features are created instead to reduce
train-serving skew. Holiday events are resolved by locale: national holidays apply
to every store, regional holidays match store state, and local holidays match city.

ZenML stores step artifacts in the active stack artifact store. In the local stack
this is MinIO (`minio_store`, `s3://zenml`). Optional explicit Parquet writes use
`S3_BUCKET` and related variables in `.env`.

## Human-readable feature artifact versions

The final feature artifact has a stable name:

```text
store_sales_features
```

When `ZENML_FEATURE_VERSION` is left empty, ZenML automatically creates versions
`1`, `2`, `3`, ... . You do not need the artifact UUID.

```python
from zenml.client import Client

# Latest
artifact = Client().get_artifact_version(
    name_id_or_prefix="store_sales_features"
)
latest = artifact.load()

# Exact version
artifact = Client().get_artifact_version(
    name_id_or_prefix="store_sales_features",
    version="2",
)
v2 = artifact.load()
```

The project helper also accepts `v1`, `v2`, ... as aliases for ZenML's numeric
auto-versions:

```python
from mlops_project.data.artifacts import load_feature_dataset

latest = load_feature_dataset()
v1 = load_feature_dataset(version="v1")  # resolves to ZenML version "1"
v2 = load_feature_dataset(version="v2")  # resolves to ZenML version "2"
```

The training loader uses the same helper. Set:

```bash
TRAIN_FEATURE_VERSION=v2
```

or leave it empty to train from the latest feature artifact.

`FEATURE_DATA_VERSION` is separate from the ZenML artifact version and controls
only the explicit MinIO path:

```text
s3://ml-data/features/store-sales/<feature-data-version>/features.parquet
```

This separation avoids coupling the physical S3 path to ZenML's artifact-version
control plane.

Copy `.env.example` to `.env` for local-only configuration. Keep credentials out
of version control.
