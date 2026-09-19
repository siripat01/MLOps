from __future__ import annotations

from typing import Annotated

import pandas as pd
from zenml import step

from mlops_project.monitoring.baseline import build_monitoring_baseline


@step(enable_cache=False)
def build_monitoring_baseline_step(
    training_data: pd.DataFrame,
) -> Annotated[dict[str, dict[str, float]], "monitoring_baseline"]:
    return build_monitoring_baseline(training_data)
