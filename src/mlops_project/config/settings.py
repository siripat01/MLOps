from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import UTC, datetime


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _optional_env(name: str) -> str | None:
    return os.getenv(name) or None


@dataclass(frozen=True)
class DataPipelineSettings:
    dataset_name: str = field(default_factory=lambda: _env("DATASET_NAME", "store-sales"))
    kaggle_competition: str = field(
        default_factory=lambda: _env(
            "KAGGLE_COMPETITION", "store-sales-time-series-forecasting"
        )
    )
    dataset_version: str = field(
        default_factory=lambda: _env(
            "DATASET_VERSION", datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        )
    )
    feature_artifact_name: str = field(
        default_factory=lambda: _env("FEATURE_ARTIFACT_NAME", "store_sales_features")
    )
    feature_version: str | None = field(default_factory=lambda: _optional_env("FEATURE_VERSION"))
    s3_endpoint_url: str = field(default_factory=lambda: _env("S3_ENDPOINT_URL", ""))
    s3_access_key: str = field(default_factory=lambda: _env("S3_ACCESS_KEY", ""))
    s3_secret_key: str = field(default_factory=lambda: _env("S3_SECRET_KEY", ""))
    s3_bucket: str = field(default_factory=lambda: _env("S3_BUCKET", ""))
    s3_region: str = field(default_factory=lambda: _env("S3_REGION", "us-east-1"))
    raw_prefix: str = field(default_factory=lambda: _env("RAW_PREFIX", "raw"))
    intermediate_prefix: str = field(
        default_factory=lambda: _env("INTERMEDIATE_PREFIX", "intermediate")
    )
    feature_prefix: str = field(default_factory=lambda: _env("FEATURE_PREFIX", "features"))

    @property
    def raw_uri_prefix(self) -> str | None:
        return self._uri(self.raw_prefix, self.dataset_name, self.dataset_version)

    @property
    def feature_uri(self) -> str | None:
        if not self.feature_version:
            return None
        return self._uri(
            self.feature_prefix, self.dataset_name, self.feature_version, "features.parquet"
        )

    def intermediate_uri(self, stage: str) -> str | None:
        return self._uri(
            self.intermediate_prefix,
            self.dataset_name,
            stage,
            self.dataset_version,
            "data.parquet",
        )

    def _uri(self, *parts: str) -> str | None:
        if not self.s3_bucket:
            return None
        cleaned = "/".join(part.strip("/") for part in parts if part)
        return f"s3://{self.s3_bucket}/{cleaned}"


def get_settings() -> DataPipelineSettings:
    return DataPipelineSettings()
