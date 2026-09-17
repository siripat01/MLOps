from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field


class HistoryPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    date: date
    sales: float
    onpromotion: float = 0
    is_holiday: bool = False
    holiday_count: int | None = None
    holiday_type: str | None = None


class KnownCovariatePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    date: date
    onpromotion: float = 0
    is_holiday: bool = False
    holiday_count: int | None = None
    holiday_type: str | None = None


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    history: list[HistoryPoint] = Field(min_length=1)
    known_covariates: list[KnownCovariatePoint] = Field(min_length=1)


class PredictionPoint(BaseModel):
    item_id: str
    timestamp: datetime
    values: dict[str, float]


class PredictionResponse(BaseModel):
    predictions: list[PredictionPoint]


class ModelMetadata(BaseModel):
    model_name: str
    model_version: str
    artifact_uri: str
    framework: str
    autogluon_version: str
    python_version: str
    artifact_sha256: str
