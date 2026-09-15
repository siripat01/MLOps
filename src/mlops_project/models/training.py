from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor

from mlops_project.data.features import KNOWN_COVARIATE_COLUMNS
from mlops_project.features.covariates import to_timeseries_frame

LOCAL_SAFE_HYPERPARAMETERS: dict[str, dict[str, Any]] = {
    "SeasonalNaive": {},
    "RecursiveTabular": {},
    "DirectTabular": {},
    "DynamicOptimizedTheta": {},
    "AutoETS": {},
}


def _hyperparameters_for_profile(model_profile: str | None) -> str | dict[str, dict[str, Any]]:
    profile = (model_profile or "default").strip().lower()
    if profile in {"default", "full", "all"}:
        return "default"
    if profile in {"local_safe", "local", "tabular"}:
        return LOCAL_SAFE_HYPERPARAMETERS
    if profile == "classical":
        return {
            "SeasonalNaive": {},
            "DynamicOptimizedTheta": {},
            "AutoETS": {},
        }
    raise ValueError(
        "Unsupported AUTOGLUON_MODEL_PROFILE "
        f"{model_profile!r}. Use one of: local_safe, classical, default."
    )


def train_store_sales_predictor(
    train_data: pd.DataFrame,
    *,
    prediction_length: int,
    presets: str = "best_quality",
    eval_metric: str = "RMSLE",
    time_limit: int | None = None,
    enable_ensemble: bool = True,
    model_profile: str | None = "local_safe",
    verbosity: int = 3,
    model_root: Path | str = "artifacts/autogluon/store_sales",
) -> tuple[Path, dict[str, Any]]:
    train_ts = to_timeseries_frame(train_data)
    model_path = Path(model_root).resolve() / uuid4().hex
    hyperparameters = _hyperparameters_for_profile(model_profile)

    predictor = TimeSeriesPredictor(
        target="sales",
        prediction_length=prediction_length,
        freq="D",
        known_covariates_names=KNOWN_COVARIATE_COLUMNS,
        eval_metric=eval_metric,
        path=str(model_path),
    )

    predictor.fit(
        train_ts,
        presets=presets,
        hyperparameters=hyperparameters,
        time_limit=time_limit,
        enable_ensemble=enable_ensemble,
        verbosity=verbosity,
    )

    fitted_models = predictor.model_names()
    if not fitted_models:
        raise RuntimeError(
            "Training did not produce a fitted model. Check the logs for CUDA "
            "or data compatibility errors."
        )

    leaderboard = predictor.leaderboard(silent=True)
    best_model = (
        predictor.model_best
        if hasattr(predictor, "model_best")
        else predictor.get_model_best()
    )
    metadata: dict[str, Any] = {
        "prediction_length": prediction_length,
        "presets": presets,
        "eval_metric": eval_metric,
        "time_limit": time_limit,
        "enable_ensemble": enable_ensemble,
        "model_profile": model_profile,
        "hyperparameters": (
            hyperparameters if isinstance(hyperparameters, str) else sorted(hyperparameters)
        ),
        "known_covariates": KNOWN_COVARIATE_COLUMNS,
        "train_rows": len(train_data),
        "train_items": int(train_data["item_id"].nunique()),
        "model_count": len(fitted_models),
        "model_names": fitted_models,
        "best_model": best_model,
        "model_path": str(model_path),
        "leaderboard": leaderboard.to_dict(orient="records"),
    }
    return model_path, metadata
