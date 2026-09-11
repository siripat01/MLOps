import polars as pl
import pytest

from mlops_project.data.artifacts import normalize_artifact_version
from mlops_project.data.cleaning import clean_raw_tables
from mlops_project.data.features import (
    HISTORICAL_COVARIATE_COLUMNS,
    KNOWN_COVARIATE_COLUMNS,
    TRAINING_COLUMNS,
    engineer_store_sales_features,
)
from mlops_project.data.integration import integrate_store_sales, scope_holidays_to_stores
from mlops_project.data.quality import validate_cleaning_row_loss
from mlops_project.data.schemas.cleaned import validate_cleaned_tables
from mlops_project.data.schemas.features import validate_feature_table
from mlops_project.data.schemas.raw import validate_raw_tables
from mlops_project.data.transformations import transform_store_sales
from pipelines.training.steps.prepare_split import prepare_training_data


def raw_tables() -> dict[str, pl.DataFrame]:
    return {
        "train": pl.DataFrame(
            {
                "id": [1, 2, 3],
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "store_nbr": [1, 1, 1],
                "family": [" grocery ", "grocery", "grocery"],
                "sales": [10.0, 20.0, 30.0],
                "onpromotion": [0, 1, 0],
            }
        ).with_columns(pl.col("date").str.to_date()),
        "stores": pl.DataFrame(
            {
                "store_nbr": [1],
                "city": ["Quito"],
                "state": ["Pichincha"],
                "type": ["D"],
                "cluster": [13],
            }
        ),
        "oil": pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "dcoilwtico": [80.0, None, 82.0],
            }
        ).with_columns(pl.col("date").str.to_date()),
        "holidays_events": pl.DataFrame(
            {
                "date": ["2024-01-02"],
                "type": ["Holiday"],
                "locale": ["National"],
                "locale_name": ["Ecuador"],
                "description": ["Test Holiday"],
                "transferred": [False],
            }
        ).with_columns(pl.col("date").str.to_date()),
        "transactions": pl.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
                "store_nbr": [1, 1, 1],
                "transactions": [100, 110, 120],
            }
        ).with_columns(pl.col("date").str.to_date()),
    }


def test_raw_schema_is_structural_and_allows_cleanable_values() -> None:
    tables = raw_tables()
    tables["train"] = tables["train"].with_columns(
        pl.when(pl.col("id") == 2).then(-1.0).otherwise(pl.col("sales")).alias("sales")
    )
    validated = validate_raw_tables(tables)
    assert validated["train"].height == 3

    cleaned, _ = clean_raw_tables(validated)
    assert cleaned["train"].height == 2


def test_cleaning_is_column_aware_and_normalizes_categories() -> None:
    tables = raw_tables()
    tables["train"] = pl.concat([tables["train"], tables["train"].head(1)])
    cleaned, metrics = clean_raw_tables(tables)
    assert cleaned["train"].height == 3
    assert cleaned["train"]["family"].to_list() == ["GROCERY", "GROCERY", "GROCERY"]
    assert metrics["train"]["rows_before"] == 4
    assert metrics["train"]["rows_removed"] == 1


def test_join_does_not_multiply_rows() -> None:
    cleaned, _ = clean_raw_tables(raw_tables())
    validated = validate_cleaned_tables(cleaned)
    integrated, metrics = integrate_store_sales(validated)
    assert integrated.height == cleaned["train"].height
    assert metrics["unmatched_stores"] == 0


def test_join_rejects_duplicate_store_dimension_keys() -> None:
    tables = raw_tables()
    tables["stores"] = pl.concat([tables["stores"], tables["stores"]])
    with pytest.raises(ValueError, match="stores must be unique"):
        integrate_store_sales(tables)


def test_cleaned_validation_rejects_missing_dates_per_series() -> None:
    tables = raw_tables()
    tables["train"] = tables["train"].filter(pl.col("date") != pl.date(2024, 1, 2))
    cleaned, _ = clean_raw_tables(tables)
    with pytest.raises(ValueError, match="missing calendar dates"):
        validate_cleaned_tables(cleaned)


def test_cleaned_validation_rejects_duplicate_store_family_date_keys() -> None:
    tables = raw_tables()
    duplicate_key = tables["train"].head(1).with_columns(pl.lit(99).cast(pl.Int64).alias("id"))
    tables["train"] = pl.concat([tables["train"], duplicate_key])
    cleaned, _ = clean_raw_tables(tables)
    with pytest.raises(ValueError, match="duplicate store/family/date"):
        validate_cleaned_tables(cleaned)


def test_cleaning_row_loss_gate_rejects_excessive_loss() -> None:
    metrics = {
        "train": {
            "rows_before": 10,
            "rows_after": 6,
            "rows_removed": 4,
        }
    }
    with pytest.raises(ValueError, match="Unexpected row loss"):
        validate_cleaning_row_loss(metrics, max_loss_rates={"train": 0.25})


def test_join_rejects_store_and_transaction_miss_thresholds() -> None:
    tables = raw_tables()
    tables["stores"] = tables["stores"].with_columns(pl.lit(2).alias("store_nbr"))
    with pytest.raises(ValueError, match="Store join produced unmatched rows"):
        integrate_store_sales(tables)

    tables = raw_tables()
    tables["transactions"] = tables["transactions"].head(1)
    with pytest.raises(ValueError, match="Transaction join miss rate exceeds threshold"):
        integrate_store_sales(
            tables,
            max_oil_miss_rate=1.0,
            max_transaction_miss_rate=0.25,
        )


def test_local_holiday_only_applies_to_matching_city() -> None:
    stores = pl.DataFrame(
        {
            "store_nbr": [1, 2],
            "city": ["QUITO", "CUENCA"],
            "state": ["PICHINCHA", "AZUAY"],
            "type": ["D", "A"],
            "cluster": [13, 1],
        }
    )
    holidays = pl.DataFrame(
        {
            "date": ["2024-01-02"],
            "type": ["HOLIDAY"],
            "locale": ["LOCAL"],
            "locale_name": ["QUITO"],
            "description": ["QUITO DAY"],
            "transferred": [False],
        }
    ).with_columns(pl.col("date").str.to_date())

    scoped = scope_holidays_to_stores(holidays, stores)
    assert scoped["store_nbr"].to_list() == [1]


def test_date_lag_and_rolling_features_use_history_only() -> None:
    cleaned, _ = clean_raw_tables(raw_tables())
    validated = validate_cleaned_tables(cleaned)
    integrated, _ = integrate_store_sales(validated)
    transformed = transform_store_sales(integrated)
    features = engineer_store_sales_features(transformed)

    assert features["month"].to_list() == [1, 1, 1]
    assert features["day_of_week"].to_list() == [1, 2, 3]
    assert features["sales_lag_1"].to_list() == [None, 10.0, 20.0]
    assert features["transactions_lag_1"].to_list() == [None, 100, 110]
    assert features["store_sales_lag_1"].to_list() == [None, 10.0, 20.0]
    assert features["family_sales_lag_1"].to_list() == [None, 10.0, 20.0]
    assert "transactions" not in features.columns
    assert "dcoilwtico" not in features.columns
    assert features["sales_rolling_mean_7"].to_list() == [None, None, None]


def test_feature_schema_accepts_engineered_features() -> None:
    cleaned, _ = clean_raw_tables(raw_tables())
    validated = validate_cleaned_tables(cleaned)
    integrated, _ = integrate_store_sales(validated)
    transformed = transform_store_sales(integrated)
    features = engineer_store_sales_features(transformed)
    validated_features = validate_feature_table(features)
    assert validated_features.height == 3


def test_training_column_selection_matches_serving_covariates() -> None:
    cleaned, _ = clean_raw_tables(raw_tables())
    validated = validate_cleaned_tables(cleaned)
    integrated, _ = integrate_store_sales(validated)
    transformed = transform_store_sales(integrated)
    features = engineer_store_sales_features(transformed)

    training_data = prepare_training_data.entrypoint(features)

    assert training_data.columns.to_list() == TRAINING_COLUMNS
    assert set(KNOWN_COVARIATE_COLUMNS).issubset(training_data.columns)
    assert set(HISTORICAL_COVARIATE_COLUMNS).isdisjoint(training_data.columns)


def test_human_artifact_version_aliases() -> None:
    assert normalize_artifact_version(None) is None
    assert normalize_artifact_version("latest") is None
    assert normalize_artifact_version(1) == "1"
    assert normalize_artifact_version("v2") == "2"
    assert normalize_artifact_version("3") == "3"
