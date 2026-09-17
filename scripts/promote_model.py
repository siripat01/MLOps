from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import boto3


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected s3://bucket/key URI, got: {uri}")
    bucket, separator, key = uri.removeprefix("s3://").partition("/")
    if not bucket or not separator or not key:
        raise ValueError(f"Expected s3://bucket/key URI, got: {uri}")
    return bucket, key


def build_pointer(
    version: str, artifact_uri: str, archive_sha256: str | None = None
) -> dict[str, str]:
    if not version or not artifact_uri:
        raise ValueError("version and artifact_uri are required")
    pointer = {"version": version, "artifact_uri": artifact_uri}
    if archive_sha256:
        pointer["archive_sha256"] = archive_sha256
    return pointer


@dataclass(frozen=True)
class PromotionConfig:
    endpoint_url: str | None = None
    access_key: str | None = None
    secret_key: str | None = None
    region: str = "us-east-1"


def promote_model(
    *,
    version: str,
    artifact_uri: str,
    pointer_uri: str,
    archive_sha256: str | None = None,
    config: PromotionConfig | None = None,
) -> None:
    config = config or PromotionConfig()
    artifact_bucket, artifact_key = parse_s3_uri(artifact_uri)
    pointer_bucket, pointer_key = parse_s3_uri(pointer_uri)
    client = boto3.client(
        "s3",
        endpoint_url=config.endpoint_url,
        aws_access_key_id=config.access_key,
        aws_secret_access_key=config.secret_key,
        region_name=config.region,
    )
    head = client.head_object(Bucket=artifact_bucket, Key=artifact_key)
    archive_sha256 = archive_sha256 or head.get("Metadata", {}).get("archive_sha256")
    client.put_object(
        Bucket=pointer_bucket,
        Key=pointer_key,
        Body=json.dumps(build_pointer(version, artifact_uri, archive_sha256), indent=2).encode(),
        ContentType="application/json",
    )


def wait_ready(base_url: str, timeout_seconds: int = 120) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = base_url.rstrip("/") + "/ready"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.HTTPError):
            time.sleep(2)
    raise TimeoutError(f"Serving endpoint did not become ready: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote or rollback an AutoGluon model pointer")
    parser.add_argument("command", choices=("promote", "rollback"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--artifact-uri", required=True)
    parser.add_argument("--pointer-uri", required=True)
    parser.add_argument("--archive-sha256")
    args = parser.parse_args()
    config = PromotionConfig(
        endpoint_url=os.getenv("S3_ENDPOINT_URL") or None,
        access_key=os.getenv("S3_ACCESS_KEY") or None,
        secret_key=os.getenv("S3_SECRET_KEY") or None,
        region=os.getenv("S3_REGION", "us-east-1"),
    )
    promote_model(
        version=args.version,
        artifact_uri=args.artifact_uri,
        pointer_uri=args.pointer_uri,
        archive_sha256=args.archive_sha256,
        config=config,
    )
    rollout_url = os.getenv("SERVING_BASE_URL")
    if rollout_url:
        wait_ready(rollout_url, int(os.getenv("READY_TIMEOUT_SECONDS", "120")))


if __name__ == "__main__":
    main()
