from datetime import date

import pandas as pd
import pytest
from pydantic import ValidationError

from mlops_project.data.features import KNOWN_COVARIATE_COLUMNS
from mlops_project.serving.forecasting import StoreSalesForecaster
from mlops_project.serving.schemas import ForecastRequest


def test_forecast_request_requires_future_known_covariates() -> None:
    with pytest.raises(ValidationError):
        ForecastRequest(
            history=[
                {
                    "item_id": "1_AUTOMOTIVE",
                    "date": date(2017, 1, 1),
                    "sales": 10,
                }
            ],
            known_covariates=[],
        )


def test_forecast_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ForecastRequest(
            history=[
                {
                    "item_id": "1_AUTOMOTIVE",
                    "date": date(2017, 1, 1),
                    "sales": 10,
                    "unexpected": "value",
                }
            ],
            known_covariates=[
                {
                    "item_id": "1_AUTOMOTIVE",
                    "date": date(2017, 1, 2),
                }
            ],
        )


def test_prediction_serialization_preserves_forecast_columns() -> None:
    index = pd.MultiIndex.from_tuples(
        [("1_AUTOMOTIVE", pd.Timestamp("2017-01-02"))],
        names=["item_id", "timestamp"],
    )
    predictions = pd.DataFrame(
        {"mean": [12.5], "0.5": [12.0]},
        index=index,
    )

    result = StoreSalesForecaster._serialize(predictions)

    assert len(result) == 1
    assert result[0].item_id == "1_AUTOMOTIVE"
    assert result[0].timestamp == pd.Timestamp("2017-01-02").to_pydatetime()
    assert result[0].values == {"mean": 12.5, "0.5": 12.0}


def test_serving_frames_derive_model_known_covariates() -> None:
    request = ForecastRequest(
        history=[
            {
                "item_id": "1_AUTOMOTIVE",
                "date": date(2017, 1, 1),
                "sales": 10,
                "onpromotion": 2,
                "is_holiday": True,
            }
        ],
        known_covariates=[
            {
                "item_id": "1_AUTOMOTIVE",
                "date": date(2017, 1, 2),
                "onpromotion": 1,
                "is_holiday": False,
            }
        ],
    )

    history = StoreSalesForecaster._history_frame(request.history)
    future = StoreSalesForecaster._known_covariates_frame(request.known_covariates)

    assert set(KNOWN_COVARIATE_COLUMNS).issubset(history.columns)
    assert set(KNOWN_COVARIATE_COLUMNS).issubset(future.columns)
    assert "sales" not in future.columns
