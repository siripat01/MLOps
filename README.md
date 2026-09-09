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
├── pipelines/            # Data, training, and release pipeline definitions
├── scripts/              # Developer and automation entry points
├── src/mlops_project/    # Reusable application and serving code
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

ZenML automatically creates versions `1`, `2`, `3`, ... for that name. You do not
need the artifact UUID.

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

## Training with Docker and GPU

Start the local control-plane services before running a pipeline:

```bash
docker compose -f infrastructure/docker/docker-compose.yml up -d
make data-pipeline
make train-pipeline
```

The `forecast-api` service is behind the Compose `serving` profile, so starting the
normal development stack does not try to pull a forecast image before one exists.

The training entrypoint uses `http://172.17.0.1:8080` by default so Docker step
containers can reach the ZenML server published on the host. Override it when
the Docker bridge gateway differs:

```bash
ZENML_DOCKER_STORE_URL=http://<docker-host-gateway>:8080 make train-pipeline
```

Set `TRAIN_FEATURE_VERSION` to a numeric version such as `4`, or an alias such
as `v4`. Leave it empty to use the latest feature artifact.

Training publishes the fitted AutoGluon predictor under the stable ZenML artifact
name configured by `MODEL_ARTIFACT_NAME` (default: `store_sales_model`). ZenML
creates a new artifact version for each training run.

Copy `.env.example` to `.env` for local-only configuration. Keep credentials out
of version control.

## Model release pipeline

Training and release are separate lifecycles. The release pipeline never silently
trains a new model. It releases an existing versioned ZenML model artifact:

```text
ZenML model artifact
    -> evaluate on the selected dataset split
    -> metric quality gate
    -> validate AutoGluon predictor
    -> register predictor in the local Bento model store
    -> build Bento from version-controlled serving source
    -> export .bento as a ZenML artifact
    -> import Bento in the image-build step
    -> build OCI image
    -> optionally push image
```

Release the latest model artifact:

```bash
MODEL_ARTIFACT_NAME=store_sales_model make deploy-build-pipeline
```

Or release an exact model version:

```bash
MODEL_ARTIFACT_NAME=store_sales_model \
MODEL_ARTIFACT_VERSION=3 \
make deploy-build-pipeline
```

The release quality gate checks model metrics such as `RMSLE`, optional `WQL`, and
optional `RMSE`. Source-code linting and unit tests are CI responsibilities and are
not executed from the ML quality-gate step.

`pipelines/deployment/steps/bento.py` packages the model but does not generate
Python source code. The actual serving implementation lives in version control:

```text
src/mlops_project/serving/
├── service.py       # BentoML transport/service boundary
├── schemas.py       # typed request/response contracts
└── forecasting.py   # AutoGluon adapter and serialization
```

The Bento itself is exported as a `.bento` ZenML `Path` artifact. This keeps
separate ZenML steps independent from a process-local Bento store.

## BentoML serving

The runtime is a native BentoML service. `StoreSalesForecastService` is defined
with `@bentoml.service` and exposes forecasting with `@bentoml.api`. There is no
FastAPI proxy, Docker SDK, generated `service.py`, or Docker-socket mount in the
inference application.

Runtime ownership is intentionally simple:

```text
Docker Compose / Kubernetes
    -> Bento OCI image
    -> BentoML Service
    -> StoreSalesForecaster
    -> AutoGluon TimeSeriesPredictor
```

After the release pipeline builds or pushes an image, configure its exact tag:

```bash
BENTO_RUNTIME_IMAGE=registry.example.com/mlops/store-sales-forecast:<version>
```

If the registry is private, authenticate Docker first. Then run the image directly:

```bash
make serve-api
curl http://localhost:8000/readyz
```

Example request:

```bash
curl -X POST http://localhost:8000/forecast \
  -H 'content-type: application/json' \
  -d '{
    "history": [
      {
        "item_id": "1_AUTOMOTIVE",
        "date": "2017-01-01",
        "sales": 10,
        "onpromotion": 0,
        "is_holiday": false
      }
    ],
    "known_covariates": [
      {
        "item_id": "1_AUTOMOTIVE",
        "date": "2017-01-02",
        "onpromotion": 0,
        "is_holiday": false
      }
    ]
  }'
```

For a real forecast, `known_covariates` must cover the model's future prediction
horizon for every requested series. Container lifecycle, image pulls, rollouts,
health checks, and restarts belong to Docker Compose/Kubernetes rather than the
Python inference code.
