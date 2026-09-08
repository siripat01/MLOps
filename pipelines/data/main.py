from __future__ import annotations

from zenml import pipeline

from pipelines.data.steps.clean import clean_data
from pipelines.data.steps.feature_engineering import engineer_features
from pipelines.data.steps.ingest import ingest_data
from pipelines.data.steps.integrate import integrate_data
from pipelines.data.steps.profile import profile_data
from pipelines.data.steps.transform import transform_data
from pipelines.data.steps.validate_features import validate_features
from pipelines.data.steps.validate_raw import validate_raw_data


@pipeline
def data_pipeline() -> None:
    raw_tables, ingestion_metadata = ingest_data()
    validated_raw, _raw_validation_metadata = validate_raw_data(raw_tables, ingestion_metadata)
    profile_data(validated_raw)
    cleaned_tables, _cleaning_metadata = clean_data(validated_raw)
    integrated_data, _join_metadata = integrate_data(cleaned_tables)
    transformed_data, _transform_metadata = transform_data(integrated_data)
    features, feature_metadata = engineer_features(transformed_data)
    validate_features(features, feature_metadata)


def main() -> None:
    data_pipeline()


if __name__ == "__main__":
    main()
