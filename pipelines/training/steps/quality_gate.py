from __future__ import annotations

from typing import Annotated, Any

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
    normalized = {
        str(key).lower(): float(value)
        for key, value in metrics.items()
        if isinstance(value, int | float)
    }
    active = {key.lower(): float(value) for key, value in thresholds.items() if value is not None}
    failures = []
    for name, threshold in active.items():
        value = normalized.get(name)
        if value is None:
            failures.append(f"{name} is missing")
        elif value > threshold:
            failures.append(f"{name}={value:.6f} exceeds {threshold:.6f}")
    if failures:
        raise RuntimeError("Quality gate failed: " + "; ".join(failures))
    return QualityGateResult(passed=True, metrics=normalized, thresholds=active)


@step(enable_cache=False)
def quality_gate(
    metrics: dict[str, float],
    max_rmsle: float = 0.75,
    max_wql: float | None = None,
    max_rmse: float | None = None,
) -> tuple[
    Annotated[bool, "passed"],
    Annotated[dict[str, float], "metrics"],
    Annotated[dict[str, float], "thresholds"],
]:
    result = assert_metric_thresholds(
        metrics,
        {"rmsle": max_rmsle, "wql": max_wql, "rmse": max_rmse},
    )
    return result.passed, result.metrics, result.thresholds
