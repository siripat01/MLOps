from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from threading import Lock

PredictionKey = tuple[str, str]
ActualPoint = tuple[str, str, float]


def compute_feedback_metrics(
    predictions: dict[PredictionKey, float],
    actuals: Iterable[ActualPoint],
) -> dict[str, float]:
    errors = [
        predictions[(item_id, timestamp)] - actual
        for item_id, timestamp, actual in actuals
        if (item_id, timestamp) in predictions
    ]
    if not errors:
        return {"matched_points": 0.0, "mae": 0.0, "rmse": 0.0}
    absolute_errors = [abs(error) for error in errors]
    return {
        "matched_points": float(len(errors)),
        "mae": sum(absolute_errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
    }


class FeedbackStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = Lock()

    def append(self, payload: dict[str, object]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, sort_keys=True) + "\n")
