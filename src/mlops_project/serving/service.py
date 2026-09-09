from __future__ import annotations

from pathlib import Path

import bentoml

from mlops_project.serving.forecasting import StoreSalesForecaster
from mlops_project.serving.schemas import ForecastRequest, ForecastResponse

BENTO_MODEL_ALIAS = "store_sales_model"

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
    name="store_sales_forecaster",
    image=service_image,
    traffic={"timeout": 120},
)
class StoreSalesForecastService:
    model_ref = bentoml.models.BentoModel(BENTO_MODEL_ALIAS)

    def __init__(self) -> None:
        model_path = Path(self.model_ref.path_of("learner.pkl")).parent
        self._forecaster = StoreSalesForecaster.load(model_path)

    @bentoml.api(route="/forecast")
    def forecast(self, request: ForecastRequest) -> ForecastResponse:
        return self._forecaster.predict(request)
