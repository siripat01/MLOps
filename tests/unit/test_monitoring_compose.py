from pathlib import Path


def test_compose_includes_monitoring_services_and_api_scrape_target() -> None:
    compose = Path("infrastructure/docker/docker-compose.yml").read_text()

    assert "prometheus:" in compose
    assert "grafana:" in compose
    assert "forecast-api:8000" in Path("infrastructure/monitoring/prometheus.yml").read_text()
    assert "MONITORING_FEEDBACK_PATH" in compose
