import json
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry

from app.feedback import FeedbackStore, compute_feedback_metrics
from app.monitoring import MonitoringRecorder, WebhookAlerter, drift_score


def test_drift_score_is_zero_for_baseline_mean() -> None:
    assert drift_score(10.0, {"mean": 10.0, "std": 2.0}) == 0.0


def test_drift_score_is_standardized_mean_shift() -> None:
    assert drift_score(14.0, {"mean": 10.0, "std": 2.0}) == 2.0


def test_monitoring_recorder_exposes_prediction_and_drift_metrics() -> None:
    registry = CollectorRegistry()
    recorder = MonitoringRecorder(registry=registry)

    recorder.record_prediction(
        model_version="v1",
        observations={"sales": 14.0},
        prediction_count=2,
        baseline={"sales": {"mean": 10.0, "std": 2.0}},
    )

    assert registry.get_sample_value(
        "forecast_api_prediction_count_total", {"model_version": "v1"}
    ) == 2.0
    assert registry.get_sample_value(
        "forecast_api_drift_score", {"model_version": "v1", "feature": "sales"}
    ) == 2.0


def test_feedback_metrics_match_prediction_means() -> None:
    predictions = {
        ("a", "2024-01-01"): 10.0,
        ("a", "2024-01-02"): 14.0,
    }

    metrics = compute_feedback_metrics(
        predictions,
        [("a", "2024-01-01", 12.0), ("a", "2024-01-02", 10.0)],
    )

    assert metrics["matched_points"] == 2.0
    assert metrics["mae"] == 3.0
    assert metrics["rmse"] == pytest.approx(3.1622776602)


def test_feedback_store_appends_jsonl_without_request_history(tmp_path: Path) -> None:
    path = tmp_path / "feedback.jsonl"
    store = FeedbackStore(path)

    store.append(
        {
            "request_id": "req-1",
            "model_version": "v1",
            "metrics": {"mae": 1.0},
        }
    )

    assert json.loads(path.read_text()) == {
        "request_id": "req-1",
        "model_version": "v1",
        "metrics": {"mae": 1.0},
    }


def test_webhook_alerter_sends_alert_without_raw_payload() -> None:
    sent: list[dict[str, object]] = []
    alerter = WebhookAlerter("https://alerts.example.test", sender=sent.append)

    alerter.notify("drift", {"model_version": "v1", "feature": "sales"}, 4.0)

    assert sent == [
        {
            "alert": "drift",
            "labels": {"model_version": "v1", "feature": "sales"},
            "value": 4.0,
        }
    ]


def test_default_monitoring_recorder_renders_default_registry() -> None:
    payload, content_type = MonitoringRecorder().render()

    assert b"forecast_api_model_ready" in payload
    assert content_type.startswith("text/plain")
