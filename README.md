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

Start the local services before running a pipeline:

```bash
docker compose -f infrastructure/docker/docker-compose.yml up -d
make data-pipeline
make train-pipeline
```

The training entrypoint uses `http://172.17.0.1:8080` by default so Docker step
containers can reach the ZenML server published on the host. Override it when
the Docker bridge gateway differs:

```bash
ZENML_DOCKER_STORE_URL=http://<docker-host-gateway>:8080 make train-pipeline
```

Set `TRAIN_FEATURE_VERSION` to a numeric version such as `4`, or an alias such
as `v4`. Leave it empty to use the latest feature artifact.

Copy `.env.example` to `.env` for local-only configuration. Keep credentials out
of version control.

## BentoML image build phase

Register a local ZenML stack with the BentoML model deployer once:

```bash
make register-bentoml-stack
```

Build a BentoML image only after the quality gate passes. By default this
pipeline trains a fresh model in the same run, then performs release packaging:

```bash
BENTO_IMAGE_TAG='mlops-project/store-sales-forecast:{version}' make deploy-build-pipeline
```

To release an already-versioned ZenML model artifact instead, set:

```bash
MODEL_ARTIFACT_NAME=<artifact-name> MODEL_ARTIFACT_VERSION=<version> make deploy-build-pipeline
```

The deployment pipeline trains and evaluates the model, checks metric thresholds
(`QUALITY_GATE_MAX_RMSLE`, plus optional WQL/RMSE thresholds), validates that the
AutoGluon predictor can be loaded, prepares an isolated Bento build context,
builds a Bento with `bentoml build -o tag`, and containerizes it with
`bentoml containerize`. Source-code checks belong in CI by default; set
`QUALITY_GATE_RUN_PROJECT_CHECKS=true` only for local development. It does not
start or deploy the API server yet.

## FastAPI serving phase

Serving is a separate process from the deployment pipeline. The deployment
pipeline only evaluates the model, builds the BentoML OCI image, and optionally
pushes it. The FastAPI server in `src/mlops_project/serving/server.py` owns the
runtime lifecycle: on startup it pulls `BENTO_RUNTIME_IMAGE`, replaces the
managed Bento container, waits for Bento's `/readyz`, and proxies inference
requests.

The generated Bento service still uses `@bentoml.service` and `@bentoml.api` for
model inference. These decorators define the service inside the image; Docker
is responsible for pulling and running that image.

Docker credentials must be available to the Docker daemon pulling the image; a
ZenML container-registry registration does not automatically authenticate the
runtime Docker daemon.

After the build pipeline prints the pushed image tag, set its exact value in
`.env`:

```bash
BENTO_RUNTIME_IMAGE=registry.example.com/mlops/store-sales-forecast:gitsha-timestamp
```

Then start the BentoML API:

```bash
docker login registry.example.com
docker compose -f infrastructure/docker/docker-compose.yml up -d forecast-api
curl http://localhost:8000/healthz
```

The FastAPI server exposes `/forecast` and forwards to the BentoML inference
endpoint:

```bash
curl -X POST http://localhost:8000/v1/forecast \
  -H 'content-type: application/json' \
  -d '{"history":[{"item_id":"1_1","date":"2017-01-01","target":10}]}'
```

Restarting `forecast-api` restarts only the serving process. Its startup hook
pulls the configured image and starts the Bento container; the deployment
pipeline is not restarted or invoked. For Kubernetes, keep the same separation
but move image pull and container lifecycle management to a Deployment/Pod.
