from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_uri: str
    model_sha256: str | None
    cache_dir: Path
    s3_endpoint_url: str | None
    s3_access_key: str | None
    s3_secret_key: str | None
    s3_region: str

    @classmethod
    def from_env(cls) -> Settings:
        model_uri = os.getenv("MODEL_URI", "").strip()
        if not model_uri:
            raise ValueError("MODEL_URI is required")
        return cls(
            model_uri=model_uri,
            model_sha256=os.getenv("MODEL_SHA256") or None,
            cache_dir=Path(os.getenv("MODEL_CACHE_DIR", "/tmp/model-cache")),
            s3_endpoint_url=os.getenv("S3_ENDPOINT_URL") or None,
            s3_access_key=os.getenv("S3_ACCESS_KEY") or None,
            s3_secret_key=os.getenv("S3_SECRET_KEY") or None,
            s3_region=os.getenv("S3_REGION", "us-east-1"),
        )
