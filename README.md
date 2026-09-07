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
uv run prepare-data
uv run pytest
```

The project pins Python 3.12 in `.python-version`; `uv` manages the interpreter,
virtual environment, project dependencies, and lockfile.

## Dataset

The raw UCI Online Retail files are stored under `data/raw/` and intentionally
ignored by Git. Run `uv run prepare-data` to create
`data/processed/daily_sales.csv`, a regular daily panel suitable for
AutoGluon's `TimeSeriesPredictor`. See `data/README.md` for provenance and data
preparation details.

Copy `.env.example` to `.env` for local-only configuration. Keep credentials out
of version control.
