import pandas as pd

from mlops_project.monitoring.baseline import build_monitoring_baseline


def test_build_monitoring_baseline_summarizes_numeric_training_features() -> None:
    frame = pd.DataFrame(
        {
            "sales": [1.0, 2.0, 3.0],
            "onpromotion": [0, 1, 2],
            "is_holiday": [False, True, False],
            "item_id": ["a", "a", "b"],
        }
    )

    baseline = build_monitoring_baseline(frame)

    assert baseline["sales"] == {
        "count": 3.0,
        "mean": 2.0,
        "std": 0.816496580927726,
        "min": 1.0,
        "max": 3.0,
    }
    assert "item_id" not in baseline
    assert baseline["is_holiday"]["mean"] == 1 / 3


def test_build_monitoring_baseline_uses_unit_std_for_constant_features() -> None:
    baseline = build_monitoring_baseline(pd.DataFrame({"sales": [4, 4, 4]}))

    assert baseline["sales"]["std"] == 1.0
