from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from urllib.request import Request as UrlRequest
from urllib.request import urlopen

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
)


def drift_score(value: float, baseline: Mapping[str, float]) -> float:
    standard_deviation = float(baseline.get("std", 1.0)) or 1.0
    return abs(value - float(baseline.get("mean", 0.0))) / standard_deviation


class WebhookAlerter:
    def __init__(
        self,
        url: str | None,
        *,
        timeout_seconds: float = 1.0,
        sender: Callable[[dict[str, object]], object] | None = None,
    ) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds
        self._sender = sender or self._send_http

    def _send_http(self, payload: dict[str, object]) -> object:
        if not self.url:
            return None
        request = UrlRequest(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds):
            return None

    def notify(self, alert: str, labels: Mapping[str, str], value: float) -> bool:
        if not self.url and self._sender == self._send_http:
            return False
        payload = {"alert": alert, "labels": dict(labels), "value": value}
        try:
            self._sender(payload)
        except Exception:
            return False
        return True


class MonitoringRecorder:
    def __init__(
        self,
        *,
        registry: CollectorRegistry | None = None,
        webhook_url: str | None = None,
        drift_threshold: float = 3.0,
        latency_threshold_seconds: float = 2.0,
    ) -> None:
        self.registry = registry or REGISTRY
        self.alerter = WebhookAlerter(webhook_url)
        self.drift_threshold = drift_threshold
        self.latency_threshold_seconds = latency_threshold_seconds
        self.requests = Counter(
            "forecast_api_requests_total",
            "HTTP requests handled by the forecast API.",
            ["route", "status", "model_version"],
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "forecast_api_request_duration_seconds",
            "HTTP request duration in seconds.",
            ["route", "status"],
            registry=self.registry,
        )
        self.prediction_count = Counter(
            "forecast_api_prediction_count_total",
            "Forecast points returned by the API.",
            ["model_version"],
            registry=self.registry,
        )
        self.drift = Gauge(
            "forecast_api_drift_score",
            "Standardized distance between serving observations and the training baseline.",
            ["model_version", "feature"],
            registry=self.registry,
        )
        self.feedback = Counter(
            "forecast_api_feedback_total",
            "Delayed feedback points accepted by the API.",
            ["model_version", "status"],
            registry=self.registry,
        )
        self.validation_errors = Counter(
            "forecast_api_validation_errors_total",
            "Validation errors returned by the API.",
            ["route"],
            registry=self.registry,
        )
        self.model_ready = Gauge(
            "forecast_api_model_ready",
            "Whether the serving model is ready.",
            registry=self.registry,
        )
        self.model_ready.set(0)
        self.alerts = Counter(
            "forecast_api_alerts_total",
            "Monitoring alerts emitted by the API.",
            ["alert"],
            registry=self.registry,
        )
        self.alert_failures = Counter(
            "forecast_api_alert_delivery_failures_total",
            "Monitoring alerts that failed to reach the configured webhook.",
            ["alert"],
            registry=self.registry,
        )

    def record_request(
        self,
        *,
        route: str,
        status: int,
        model_version: str,
        duration_seconds: float,
    ) -> None:
        status_class = f"{status // 100}xx"
        self.requests.labels(route, status_class, model_version).inc()
        self.request_duration.labels(route, status_class).observe(duration_seconds)
        if status >= 500:
            self._notify(
                "http_5xx",
                {"route": route, "model_version": model_version},
                float(status),
            )
        if duration_seconds >= self.latency_threshold_seconds:
            self._notify(
                "latency",
                {"route": route, "model_version": model_version},
                duration_seconds,
            )

    def record_prediction(
        self,
        *,
        model_version: str,
        observations: Mapping[str, float],
        prediction_count: int,
        baseline: Mapping[str, Mapping[str, float]],
    ) -> None:
        self.prediction_count.labels(model_version).inc(prediction_count)
        for feature, value in observations.items():
            if feature in baseline:
                score = drift_score(value, baseline[feature])
                self.drift.labels(model_version, feature).set(score)
                if score >= self.drift_threshold:
                    self._notify(
                        "drift",
                        {"model_version": model_version, "feature": feature},
                        score,
                    )

    def record_feedback(self, *, model_version: str, status: str, count: int = 1) -> None:
        self.feedback.labels(model_version, status).inc(count)

    def record_validation_error(self, *, route: str) -> None:
        self.validation_errors.labels(route).inc()

    def set_model_ready(self, ready: bool) -> None:
        self.model_ready.set(1 if ready else 0)

    def _notify(self, alert: str, labels: Mapping[str, str], value: float) -> None:
        self.alerts.labels(alert).inc()
        if not self.alerter.notify(alert, labels, value):
            self.alert_failures.labels(alert).inc()

    def render(self) -> tuple[bytes, str]:
        from prometheus_client import generate_latest

        return generate_latest(self.registry), CONTENT_TYPE_LATEST


def request_observations(request: object) -> dict[str, float]:
    history = getattr(request, "history", [])
    if not history:
        return {}
    sales = [float(point.sales) for point in history]
    promotions = [float(point.onpromotion) for point in history]
    holidays = [1.0 if point.is_holiday else 0.0 for point in history]
    return {
        "sales": sum(sales) / len(sales),
        "onpromotion": sum(promotions) / len(promotions),
        "is_holiday": sum(holidays) / len(holidays),
    }
