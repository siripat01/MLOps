import json

import pytest

from scripts.promote_model import build_pointer, parse_s3_uri, promote_model


def test_build_pointer() -> None:
    assert build_pointer("v19", "s3://ml-models/store-sales/v19/model.tar.gz", "a" * 64) == {
        "version": "v19",
        "artifact_uri": "s3://ml-models/store-sales/v19/model.tar.gz",
        "archive_sha256": "a" * 64,
    }


def test_parse_s3_uri_rejects_invalid_value() -> None:
    with pytest.raises(ValueError):
        parse_s3_uri("file:///tmp/model")


def test_promote_model_validates_and_writes_pointer(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str, bytes]] = []

    class FakeClient:
        def head_object(self, **kwargs) -> None:
            calls.append(("head", kwargs["Bucket"], kwargs["Key"].encode()))

        def put_object(self, **kwargs) -> None:
            calls.append(("put", kwargs["Bucket"], kwargs["Body"]))

    monkeypatch.setattr(
        "scripts.promote_model.boto3.client", lambda *_args, **_kwargs: FakeClient()
    )
    promote_model(
        version="v19",
        artifact_uri="s3://ml-models/store-sales/v19/model.tar.gz",
        pointer_uri="s3://ml-models/store-sales/production.json",
        archive_sha256="a" * 64,
    )

    assert calls[0] == ("head", "ml-models", b"store-sales/v19/model.tar.gz")
    assert json.loads(calls[1][2]) == build_pointer(
        "v19", "s3://ml-models/store-sales/v19/model.tar.gz", "a" * 64
    )
