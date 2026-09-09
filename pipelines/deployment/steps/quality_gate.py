from __future__ import annotations

import subprocess
from typing import Any

from pydantic import BaseModel, ConfigDict
from zenml import step


class QualityGateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    metrics: dict[str, float]
    thresholds: dict[str, float]


def assert_metric_thresholds(
    metrics: dict[str, Any],
    thresholds: dict[str, float | None],
) -> QualityGateResult:
    normalized_metrics = {
        str(key).lower(): float(value)
        for key, value in metrics.items()
        if isinstance(value, int | float)
    }
    active_thresholds = {
        key.lower(): float(value) for key, value in thresholds.items() if value is not None
    }

    failures = []
    for metric_name, threshold in active_thresholds.items():
        value = normalized_metrics.get(metric_name)
        if value is None:
            failures.append(f"{metric_name} is missing")
        elif value > threshold:
            failures.append(f"{metric_name}={value:.6f} exceeds {threshold:.6f}")

    if failures:
        raise RuntimeError("Quality gate failed: " + "; ".join(failures))

    return QualityGateResult(
        passed=True,
        metrics=normalized_metrics,
        thresholds=active_thresholds,
    )


def run_quality_commands() -> None:
    commands = [
        ["uv", "run", "ruff", "check", "."],
        ["uv", "run", "pytest", "--cov=mlops_project", "--cov-report=term-missing"],
    ]
    for command in commands:
        subprocess.run(command, check=True)


@step(enable_cache=False)
def quality_gate(
    metrics: dict[str, float],
    max_rmsle: float = 0.75,
    max_wql: float | None = None,
    max_rmse: float | None = None,
    run_project_checks: bool = False,
) -> QualityGateResult:
    gate = assert_metric_thresholds(
        metrics=metrics,
        thresholds={
            "rmsle": max_rmsle,
            "wql": max_wql,
            "rmse": max_rmse,
        },
    )
    if run_project_checks:
        run_quality_commands()
    return gate
