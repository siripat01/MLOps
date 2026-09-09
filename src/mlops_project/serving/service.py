from __future__ import annotations

import os
from pathlib import Path

import bentoml
from dotenv import load_dotenv

from mlops_project.serving.forecasting import StoreSalesForecaster
from mlops_project.serving.schemas import (
    ForecastRequest,
    ForecastResponse,
    HistoryPoint,
    KnownCovariatePoint,
)

load_dotenv()

BENTO_NAME = os.getenv("BENTO_NAME", "store_sales_forecaster")
BENTO_MODEL_ALIAS = os.getenv("BENTO_MODEL_ALIAS", "store_sales_model")

service_image = (
    bentoml.images.Image(python_version="3.12")
    .python_packages(
        "autogluon.timeseries==1.6.1",
        "pandas>=2.0,<2.4",
        "pydantic>=2,<3",
        "torch>=2.8,<2.11",
    )
)


@bentoml.service(
    name=BENTO_NAME,
    image=service_image,
    envs=[
        {"name": "BENTO_NAME", "value": BENTO_NAME},
        {"name": "BENTO_MODEL_ALIAS", "value": BENTO_MODEL_ALIAS},
    ],
    traffic={"timeout": 120},
)
class StoreSalesForecastService:
    model_ref = bentoml.models.BentoModel(BENTO_MODEL_ALIAS)

    def __init__(self) -> None:
        model_path = Path(self.model_ref.path_of("learner.pkl")).parent
        self._forecaster = StoreSalesForecaster.load(model_path)

    @bentoml.api(route="/forecast")
    def forecast(
        self,
        history: list[HistoryPoint],
        known_covariates: list[KnownCovariatePoint],
    ) -> ForecastResponse:
        request = ForecastRequest(
            history=history,
            known_covariates=known_covariates,
        )
        return self._forecaster.predict(request)
