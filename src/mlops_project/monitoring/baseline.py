from __future__ import annotations

import pandas as pd


def build_monitoring_baseline(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...] | None = None,
) -> dict[str, dict[str, float]]:
    """Build compact numeric summary statistics for serving-time drift checks."""
    selected = columns or tuple(frame.select_dtypes(include=["number", "bool"]).columns)
    baseline: dict[str, dict[str, float]] = {}
    for column in selected:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            continue
        standard_deviation = float(values.std(ddof=0))
        baseline[column] = {
            "count": float(len(values)),
            "mean": float(values.mean()),
            "std": standard_deviation or 1.0,
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return baseline
