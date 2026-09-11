from pathlib import Path

import pandas as pd
import polars as pl

from mlops_project.data import artifacts
from mlops_project.data.features import KNOWN_COVARIATE_COLUMNS
from mlops_project.features.covariates import training_frame_from_features
from mlops_project.models import eval as model_eval
from mlops_project.models import training as model_training


def training_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item_id": ["1_AUTOMOTIVE", "1_AUTOMOTIVE", "1_AUTOMOTIVE"],
            "date": pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-03"]),
            "sales": [10.0, 12.0, 11.0],
            "onpromotion": [0, 1, 0],
            "is_holiday": [False, True, False],
        }
    )


def test_training_frame_from_features_derives_all_known_covariates() -> None:
    prepared = training_frame_from_features(training_rows())

    assert set(KNOWN_COVARIATE_COLUMNS).issubset(prepared.columns)
    assert prepared["promotion_log1p"].iloc[1] > 0
    assert prepared["holiday_type_HOLIDAY"].iloc[1]


def test_direct_feature_uri_loader_reads_parquet(tmp_path) -> None:
    feature_path = tmp_path / "features.parquet"
    pl.DataFrame(training_rows()).write_parquet(feature_path)

    loaded = artifacts.load_feature_dataset_from_uri(str(feature_path))

    assert loaded.shape == (3, 5)
    assert loaded["item_id"].to_list() == ["1_AUTOMOTIVE"] * 3


def test_train_store_sales_predictor_returns_metadata(monkeypatch) -> None:
    class FakePredictor:
        model_best = "WeightedEnsemble"

        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def fit(self, train_ts, **kwargs) -> None:
            self.fit_kwargs = kwargs

        def model_names(self) -> list[str]:
            return ["Naive", "WeightedEnsemble"]

        def leaderboard(self, silent: bool = True) -> pd.DataFrame:
            return pd.DataFrame({"model": ["WeightedEnsemble"], "score_val": [-0.1]})

    monkeypatch.setattr(model_training, "TimeSeriesPredictor", FakePredictor)

    model_path, metadata = model_training.train_store_sales_predictor(
        training_rows(),
        prediction_length=1,
        presets="fast_training",
        model_root=Path("artifacts/test-autogluon"),
    )

    assert model_path.name
    assert metadata["best_model"] == "WeightedEnsemble"
    assert metadata["known_covariates"] == KNOWN_COVARIATE_COLUMNS
    assert metadata["model_count"] == 2


def test_evaluate_store_sales_predictor_returns_normalized_report(monkeypatch) -> None:
    class FakePredictor:
        @classmethod
        def load(cls, model_path: str) -> "FakePredictor":
            return cls()

        def evaluate(self, validation_ts, metrics, display: bool = True) -> dict[str, float]:
            return {"RMSLE": -0.25, "RMSE": -1.5}

        def leaderboard(self, validation_ts, silent: bool = True) -> pd.DataFrame:
            return pd.DataFrame({"model": ["Naive"], "score_test": [-0.25]})

    monkeypatch.setattr(model_eval, "TimeSeriesPredictor", FakePredictor)

    report = model_eval.evaluate_store_sales_predictor(
        Path("model"),
        training_rows().head(2),
        training_rows().tail(1),
        metrics=("RMSLE", "RMSE"),
    )

    assert report["metrics"] == {"RMSLE": -0.25, "RMSE": -1.5}
    assert report["validation_rows"] == 1
    assert report["leaderboard"][0]["model"] == "Naive"
