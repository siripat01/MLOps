from pathlib import Path

import pytest

from pipelines.deployment.steps import bento


def test_resolve_image_tag_uses_version_template() -> None:
    assert (
        bento.resolve_image_tag(
            "registry.example.com/ml/store-sales:{version}",
            version="abc123-100",
            git_sha="abc123",
        )
        == "registry.example.com/ml/store-sales:abc123-100"
    )


def test_resolve_image_tag_appends_version_when_tag_is_omitted() -> None:
    assert (
        bento.resolve_image_tag(
            "registry.example.com/ml/store-sales",
            version="abc123-100",
            git_sha="abc123",
        )
        == "registry.example.com/ml/store-sales:abc123-100"
    )


def test_resolve_image_tag_respects_explicit_tag() -> None:
    assert (
        bento.resolve_image_tag(
            "registry.example.com/ml/store-sales:staging",
            version="abc123-100",
            git_sha="abc123",
        )
        == "registry.example.com/ml/store-sales:staging"
    )


def test_parse_bento_tag_handles_bentoml_tag_output() -> None:
    assert (
        bento.parse_bento_tag("__tag__:store_sales_forecaster:abc123-100\n")
        == "store_sales_forecaster:abc123-100"
    )


def test_validate_autogluon_predictor_reports_missing_files(tmp_path: Path) -> None:
    (tmp_path / "learner.pkl").touch()

    with pytest.raises(FileNotFoundError, match="models/trainer.pkl"):
        bento.validate_autogluon_predictor(tmp_path)


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

    monkeypatch.setattr(bento.TimeSeriesPredictor, "load", fake_load)

    bento.validate_autogluon_predictor(tmp_path)

    assert loaded_paths == [str(tmp_path)]


def test_write_bento_context_packages_service_and_model(tmp_path: Path) -> None:
    model_path = tmp_path / "predictor"
    model_path.mkdir()
    (model_path / "learner.pkl").write_text("model", encoding="utf-8")
    (model_path / "models").mkdir()
    (model_path / "models" / "trainer.pkl").write_text("trainer", encoding="utf-8")

    build_context = tmp_path / "bento-context"

    bento.write_bento_context(build_context, model_path)

    assert (build_context / "service.py").exists()
    assert (build_context / "bentofile.yaml").exists()
    assert (build_context / "model" / "learner.pkl").exists()
    assert (build_context / "model" / "models" / "trainer.pkl").exists()
    assert 'Path(__file__).parent / "model"' in (build_context / "service.py").read_text(
        encoding="utf-8"
    )
