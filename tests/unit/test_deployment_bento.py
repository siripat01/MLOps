from pathlib import Path

import pytest

from pipelines.deployment.steps import bento, image, model_validation


def test_make_build_version_is_deterministic_with_explicit_timestamp() -> None:
    assert bento.make_build_version("abc123", timestamp=100) == "abc123-100"


def test_resolve_image_tag_uses_version_template() -> None:
    assert (
        image.resolve_image_tag(
            "registry.example.com/ml/store-sales:{version}",
            version="abc123-100",
            git_sha="abc123",
        )
        == "registry.example.com/ml/store-sales:abc123-100"
    )


def test_resolve_image_tag_appends_version_when_tag_is_omitted() -> None:
    assert (
        image.resolve_image_tag(
            "registry.example.com/ml/store-sales",
            version="abc123-100",
            git_sha="abc123",
        )
        == "registry.example.com/ml/store-sales:abc123-100"
    )


def test_resolve_image_tag_respects_explicit_tag() -> None:
    assert (
        image.resolve_image_tag(
            "registry.example.com/ml/store-sales:staging",
            version="abc123-100",
            git_sha="abc123",
        )
        == "registry.example.com/ml/store-sales:staging"
    )


def test_validate_autogluon_predictor_reports_missing_files(tmp_path: Path) -> None:
    (tmp_path / "learner.pkl").touch()

    with pytest.raises(FileNotFoundError, match="models/trainer.pkl"):
        model_validation.validate_autogluon_predictor(tmp_path)


def test_validate_autogluon_predictor_loads_valid_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "learner.pkl").touch()
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "trainer.pkl").touch()

    loaded_paths = []

    def fake_load(model_path: str) -> object:
        loaded_paths.append(model_path)
        return object()

    monkeypatch.setattr(model_validation.TimeSeriesPredictor, "load", fake_load)

    model_validation.validate_autogluon_predictor(tmp_path)

    assert loaded_paths == [str(tmp_path)]
