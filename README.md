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
├── pipelines/            # Training and deployment pipeline definitions
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

## Dataset

The active data source is Kaggle's Store Sales time-series competition. The
data pipeline ingests the Kaggle CSV files, validates raw contracts, profiles
the inputs, cleans and joins the tables, transforms them into a canonical
store-family daily panel, engineers historical/date/business features, and
validates the final feature dataset.

ZenML stores step artifacts in the active stack artifact store. In the local
stack this is MinIO (`minio_store`, `s3://zenml`). Optional explicit versioned
Parquet writes can also be enabled with `S3_BUCKET` and related variables in
`.env`.

Run the feature pipeline:

```bash
uv run run-data-pipeline
```

Pipeline stages:

```text
ingest_data
validate_raw_data
profile_data
clean_data
integrate_data
transform_data
engineer_features
validate_features
```

Copy `.env.example` to `.env` for local-only configuration. Keep credentials out
of version control.
